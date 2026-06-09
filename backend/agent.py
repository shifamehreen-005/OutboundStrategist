"""The agent: an explicit plan -> retrieve -> reason -> draft pipeline.

Each public method is a generator that yields progress events (so the UI
can show the agent working) and finishes by yielding a final event whose
`data` is the structured result.

The pipeline is deliberately step-wise rather than one mega-prompt:
  Mode 1 (sender):  plan pages -> fetch -> chunk -> retrieve -> synthesize ICP
  Mode 2 (target):  fetch -> web search (Tavily/native) -> chunk -> retrieve ->
                    signals + fit -> strategist -> drafter -> critic -> claim map

Token economy: BM25 does recall for free; Claude only ever sees the top-k
retrieved snippets for each sub-question, never whole pages.
"""
from __future__ import annotations

import json
import re
from typing import Dict, Iterator, List, Optional

from .config import config
from .fetcher import (fetch_site, fetch_enrichment,
                      find_customer_links, enrich_customer_snippets, Page)
from .scoring import score_fit
from .chunker import chunk_pages
from .retriever import Retriever
from .llm import LLM
from . import prompts
from .schemas import (
    Chunk, Evidence, ICP, SizeBand, Trigger, Buyer, SenderProfile, NamedCustomer,
    Signal, FitScore, DimensionJudgment, AntiPatternHit, QuestionAnswer,
    AnglePlan, CriticVerdict, Claim, Email, TargetReport,
)


# Corpus too thin to trust the synthesis at all → confidence='low', skip replan.
_THIN_CORPUS_CHUNKS = 2
_THIN_CORPUS_CHARS  = 400

# Thresholds for triggering the adaptive re-plan (profile-level, not corpus-level).
_THIN_EVIDENCE = 4   # fewer retrieved chunks → BM25 likely missed the site's language
_SHORT_VP = 60       # chars; very short value prop signals low confidence



def _evt(step: str, status: str, detail: str = "", data=None) -> Dict:
    return {"step": step, "status": status, "detail": detail, "data": data}


# URL path keywords that mark a page as a likely customer / case-study page —
# where logo walls and named customers actually live.
_CUSTOMER_PAGE_KW = ("customer", "case", "stories", "story", "testimonial", "client")


def _is_customer_chunk(c: Chunk) -> bool:
    u = c.url.lower()
    return any(k in u for k in _CUSTOMER_PAGE_KW)


def _norm_name(s: str) -> str:
    """Normalise a company name for dedup across sources (SumUp == sumup)."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


# Marketing / listicle URLs that name companies as EXAMPLES, not customers.
# A customer sourced only to such a page is almost always a false positive.
_MARKETING_URL_RE = re.compile(
    r"/blog/.*(example|guide|how-to|how_to|tips|best-|top-|template|what-is|vs-)",
    re.I,
)


def _is_marketing_url(url: str) -> bool:
    return bool(_MARKETING_URL_RE.search(url or ""))


# Normalised name substrings that reliably indicate a non-customer entry.
# Precision > recall: better to drop a borderline name than to list a vendor /
# consent tool / press mention as a customer.
_NON_CUSTOMER_NAME_RE = re.compile(
    r"\b(cookie|consent|gdpr|ccpa|onetrust|cookiebot|osano|termly|iubenda"
    r"|cloudflare|hubspot|salesforce|intercom|zendesk|stripe|twilio|sendgrid"
    r"|mailchimp|segment|mixpanel|amplitude|datadog|pagerduty|newrelic"
    r"|trustpilot|g2|capterra|gartner|forrester|techcrunch|crunchbase"
    r"|linkedin|twitter|facebook|instagram|youtube|github|gitlab"
    r"|aws|azure|gcp|google cloud|microsoft|oracle|sap)\b",
    re.I,
)

# Source-URL path fragments that signal press / partner / competitor context.
_NON_CUSTOMER_URL_RE = re.compile(
    r"/(press|partner|partners|integration|integrations|competitor|alternatives"
    r"|vs|versus|compare|comparison|award|awards|media|news|blog/[^/]*-vs-)/",
    re.I,
)


def _is_non_customer(name: str, source_url: str = "") -> bool:
    """Return True if this name is almost certainly NOT a real customer."""
    if _NON_CUSTOMER_NAME_RE.search(name or ""):
        return True
    if source_url and _NON_CUSTOMER_URL_RE.search(source_url):
        return True
    return False


def _merge_customers(groups: List[List[NamedCustomer]]) -> List[NamedCustomer]:
    """Merge customer candidates from multiple sources, deduped by normalised
    name. First occurrence wins; a later source backfills a missing source_url."""
    out: List[NamedCustomer] = []
    by_key: Dict[str, NamedCustomer] = {}
    for group in groups:
        for c in group:
            key = _norm_name(c.name)
            if not key:
                continue
            if key in by_key:
                existing = by_key[key]
                if not existing.source_url and c.source_url:
                    existing.source_url = c.source_url
                    existing.provenance = existing.provenance or c.provenance
                continue
            by_key[key] = c
            out.append(c)
    return out


def _apply_budget(chunks: List[Chunk]) -> List[Chunk]:
    """Keep chunks (in order) until the per-snippet char total would exceed the
    evidence budget. Order matters: pass the highest-priority chunks first."""
    if not config.EVIDENCE_BUDGET_CHARS:
        return chunks
    out: List[Chunk] = []
    total = 0
    cap = config.EVIDENCE_SNIPPET_CHARS
    for c in chunks:
        cost = min(len(c.text), cap) if cap else len(c.text)
        if total + cost <= config.EVIDENCE_BUDGET_CHARS:
            out.append(c)
            total += cost
    return out


def _build_overview(pages: List[Page], max_chars: int = 1200) -> str:
    """Compact site map for the adaptive planner — titles + homepage excerpt (~300 tokens)."""
    lines = ["Pages found:"]
    for p in pages:
        label = p.title or p.url.split("/")[-1] or "home"
        lines.append(f"  - {label} ({p.url})")
    if pages:
        lines.append("\nHomepage excerpt:")
        lines.append(pages[0].text[:max_chars].strip())
    return "\n".join(lines)


def _detect_gaps(retrieved: List[Chunk]) -> List[str]:
    """Keyword-scan retrieved chunks for ICP dimensions with no coverage."""
    text = " ".join(c.text.lower() for c in retrieved)
    gaps = []
    if not any(w in text for w in ["price", "pricing", "plan", "cost", "paid", "free", "tier"]):
        gaps.append("pricing model / plan tiers")
    if not any(w in text for w in ["customer", "client", "industry", "industries", "market", "serve"]):
        gaps.append("target industries / customers")
    if not any(w in text for w in ["team", "employee", "headcount", "people", "size", "scale"]):
        gaps.append("company size or segment")
    if not any(w in text for w in ["vp", "director", "head of", "manager", "ceo", "cto", "buyer"]):
        gaps.append("buyer personas")
    return gaps


def _build_profile(raw: Dict, domain: str, id_map: Dict[str, Chunk],
                   confidence: str = "high",
                   extra_gaps: Optional[List[str]] = None,
                   evidence_sufficient: bool = True,
                   named_customers: Optional[List[NamedCustomer]] = None) -> SenderProfile:
    """Construct a SenderProfile, verifying per-claim evidence_ids against id_map."""
    icp_raw = raw.get("icp", {})

    # Build triggers with grounding verification.
    triggers: List[Trigger] = []
    for t in icp_raw.get("triggers", []):
        eids = t.get("evidence_ids", [])
        grounded = bool(eids) and all(eid in id_map for eid in eids)
        triggers.append(Trigger(
            name=t.get("name", ""),
            description=t.get("description", ""),
            evidence_ids=eids,
            grounded=grounded,
        ))

    # Build buyers with grounding verification.
    buyers: List[Buyer] = []
    for b in icp_raw.get("buyers", []):
        eids = b.get("evidence_ids", [])
        grounded = bool(eids) and all(eid in id_map for eid in eids)
        buyers.append(Buyer(
            role=b.get("role", ""),
            seniority=b.get("seniority", ""),
            why=b.get("why", ""),
            evidence_ids=eids,
            grounded=grounded,
        ))

    # grounding_coverage = grounded claims / total claims across triggers + buyers.
    all_claims: List = triggers + buyers  # type: ignore[assignment]
    grounded_count = sum(1 for c in all_claims if c.grounded)
    coverage = grounded_count / len(all_claims) if all_claims else 1.0

    icp = ICP(
        industries=icp_raw.get("industries", []),
        size_bands=[SizeBand(**s) for s in icp_raw.get("size_bands", [])],
        triggers=triggers,
        buyers=buyers,
        geographies=icp_raw.get("geographies", []),
        anti_patterns=icp_raw.get("anti_patterns", []),
        qualifying_questions=icp_raw.get("qualifying_questions", []),
    )
    # Agent-forced 'low' always wins; otherwise trust the LLM's self-assessment.
    resolved_confidence = "low" if confidence == "low" else raw.get("confidence", "high")
    # Merge agent-detected gaps with any additional ones the model flagged.
    all_gaps = list({g for g in (extra_gaps or []) + raw.get("data_gaps", [])})
    # Union all cited ids so every trigger/buyer citation has an evidence panel entry.
    top_ids = raw.get("evidence_ids", [])
    facet_ids = [eid for t in triggers for eid in t.evidence_ids] + \
                [eid for b in buyers  for eid in b.evidence_ids]
    all_cited = list(dict.fromkeys(top_ids + facet_ids))  # ordered dedup
    return SenderProfile(
        company=domain,
        value_proposition=raw.get("value_proposition", ""),
        one_liner=raw.get("one_liner", ""),
        capabilities=raw.get("capabilities", []),
        icp=icp,
        named_customers=named_customers or [],
        customer_pattern=raw.get("customer_pattern", ""),
        evidence=_evidence_from_ids(all_cited, id_map),
        evidence_sufficient=evidence_sufficient,
        data_gaps=all_gaps,
        confidence=resolved_confidence,
        grounding_coverage=coverage,
    )


def _is_low_confidence(profile: SenderProfile, retrieved: List[Chunk]) -> bool:
    """Return True when the first-pass result warrants an adaptive re-plan."""
    if len(retrieved) < _THIN_EVIDENCE:
        return True
    if len(profile.value_proposition) < _SHORT_VP:
        return True
    if not profile.icp.industries:
        return True
    return False


def _evidence_from_ids(eids: List[str], id_map: Dict[str, Chunk]) -> List[Evidence]:
    out, seen = [], set()
    for eid in eids:
        c = id_map.get(eid)
        if not c or c.id in seen:
            continue
        seen.add(c.id)
        snippet = c.text.strip().replace("\n", " ")
        if len(snippet) > config.MAX_SNIPPET_CHARS:
            snippet = snippet[: config.MAX_SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"
        out.append(Evidence(id=eid, url=c.url, snippet=snippet))
    return out


class Agent:
    def __init__(self) -> None:
        self.llm = LLM()

    # -- customer discovery via web (primary: Claude search; fallback: DDG) --
    def _web_customers(self, domain: str) -> List[NamedCustomer]:
        """Find customers across the open web. Tries Claude's native web search;
        on any failure falls back to DuckDuckGo + extraction. Returns [] if both
        are unavailable — never raises, never invents."""
        # Primary: Claude native web search (highest quality, citations built in).
        try:
            raw = self.llm.web_search_json(
                prompts.CUSTOMERS_WEBSEARCH_SYSTEM,
                prompts.customers_websearch_user(domain),
                task="sender_customers_web",
                max_tokens=2500,
                max_searches=config.CUSTOMER_WEB_MAX_SEARCHES,
                # Strong model here on purpose: web-search recall/precision is the
                # proof point (finds Quora etc.) and this runs once per Mode 1.
                model=config.MODEL,
                queries=prompts.customers_websearch_queries(domain),
            )
            cust = [
                NamedCustomer(name=(c.get("name") or "").strip(),
                              source_url=(c.get("source_url") or "").strip(),
                              provenance="web")
                for c in raw.get("customers", [])
                if (c.get("name") or "").strip()
                and not _is_marketing_url(c.get("source_url") or "")
                and not _is_non_customer(
                    (c.get("name") or "").strip(),
                    (c.get("source_url") or "").strip(),
                )
            ]
            if cust:
                return cust
        except Exception:  # noqa: BLE001 — web search unsupported / errored
            pass
        # Fallback: DuckDuckGo results → LLM extraction.
        try:
            from . import search as web
            results = web.ddg_search(f"{domain} customers case study clients", limit=6)
            if not results:
                return []
            snippets = "\n".join(
                f"- {r.title} | {r.url}\n  {r.snippet}" for r in results)
            raw = self.llm.json(
                prompts.CUSTOMERS_WEBSEARCH_SYSTEM,
                f"From these search results about {domain}, list its named "
                f"customers. JSON: {{\"customers\":[{{\"name\":\"\",\"source_url\":\"\"}}]}} "
                f"— only companies explicitly called a customer of {domain}; use "
                f"the result URL as source_url.\n\n{snippets}",
                task="sender_customers_web", max_tokens=600, model=config.MODEL_LIGHT,
            )
            return [
                NamedCustomer(name=(c.get("name") or "").strip(),
                              source_url=(c.get("source_url") or "").strip(),
                              provenance="web")
                for c in raw.get("customers", [])
                if (c.get("name") or "").strip()
                and not _is_marketing_url(c.get("source_url") or "")
                and not _is_non_customer(
                    (c.get("name") or "").strip(),
                    (c.get("source_url") or "").strip(),
                )
            ]
        except Exception:  # noqa: BLE001
            return []

    # ------------------------------------------------------------------ #
    # MODE 1 — sender ICP + value proposition
    # ------------------------------------------------------------------ #
    def run_sender(self, url: str, max_pages: int) -> Iterator[Dict]:
        yield _evt("plan", "running",
                   "Planning which public pages to read (product, pricing, "
                   "customers, about, news).")

        _SENDER_COVERAGE_QUERIES = [
            "what we do product value proposition",
            "who we serve customers industries",
            "pricing plans",
            "buyers sales marketing roles",
            "results outcomes ROI",
        ]
        yield _evt("fetch", "running", f"Fetching up to {max_pages} pages from {url}…")
        pages = fetch_site(url, max_pages=max_pages,
                           coverage_queries=_SENDER_COVERAGE_QUERIES)
        yield _evt("fetch", "done",
                   f"Fetched {len(pages)} pages: "
                   + ", ".join(p.url.split('/')[-1] or 'home' for p in pages))

        domain = url.replace("https://", "").replace("http://", "").strip("/")

        # --- Thin-site enrichment: YC / Product Hunt + web search -------------
        # If the crawled site has little text, pull external sources before
        # building the BM25 index so the ICP synthesis has more to work from.
        site_chars = sum(len(p.text) for p in pages)
        if site_chars < config.ENRICH_THRESHOLD:
            # Step 1: YC + Product Hunt static pages (reuse Mode 2's helper).
            enrich_pages = fetch_enrichment(domain)
            if enrich_pages:
                pages = pages + enrich_pages
                src_labels = ", ".join(p.url.split("/")[2] for p in enrich_pages)
                yield _evt("fetch", "done",
                           f"Site thin ({site_chars} chars) — enriched from: "
                           f"{src_labels}.")

            # Step 2: web search for general positioning / funding / reviews.
            if not config.MOCK_MODE:
                try:
                    web_raw = self.llm.web_search_json(
                        prompts.SENDER_WEBSEARCH_SYSTEM,
                        prompts.sender_websearch_user(domain),
                        task="sender_web_research",
                        max_tokens=2000,
                        max_searches=3,
                        model=config.MODEL_LIGHT,
                        queries=prompts.sender_websearch_queries(domain),
                    )
                    summary = (web_raw.get("summary") or "").strip()
                    snippets = web_raw.get("snippets") or []
                    if summary or snippets:
                        web_pages = [Page(
                            url=s.get("source_url", domain),
                            title="Web research",
                            text=s.get("text", ""),
                        ) for s in snippets if s.get("text")]
                        if web_pages:
                            pages = pages + web_pages
                        yield _evt("fetch", "done",
                                   f"Web research: {len(snippets)} snippet(s) "
                                   f"added to corpus.")
                except Exception:  # noqa: BLE001 — web search optional
                    pass
            else:
                # MOCK: inject a short representative snippet so the path is
                # exercised offline without hitting the network.
                pages = pages + [Page(
                    url=f"https://mock-web-research/{domain}",
                    title="Web research (mock)",
                    text=(
                        f"{domain} is a B2B SaaS platform for outbound sales automation. "
                        "Primarily serves sales teams at Series A–C startups. "
                        "Recently raised funding and is expanding its GTM team."
                    ),
                )]

        chunks = chunk_pages(pages)
        retriever = Retriever(chunks)
        yield _evt("index", "done", f"Indexed {len(chunks)} passages (BM25).")

        # --- First pass: hardcoded queries — always free (pure BM25) ----------
        _DEFAULT_QUERIES = [
            "what we do product value proposition platform",
            "who we serve customers industries use cases",
            "pricing plans startup smb enterprise segment",
            "results outcomes ROI customer stories testimonials",
            "buyers sales marketing revenue teams roles",
        ]
        # Two separate retrievals — keeps evidence for synthesis clean (bounded
        # E-ids) while giving the customer extractor its own dedicated pool.
        _CUSTOMER_QUERIES = [
            "trusted by customers logos clients brands companies",
            "case study customer story testimonial success",
            "companies that use our product join customers",
        ]

        # Synthesis evidence — 5 ICP-focused queries, bounded budget → E1..E~6
        retrieved = retriever.multi_search(
            _DEFAULT_QUERIES, per_query=2,
            budget_chars=config.EVIDENCE_BUDGET_CHARS,
        )
        # Customer evidence — separate pool, NOT added to E-ids
        cust_retrieved = retriever.multi_search(
            _CUSTOMER_QUERIES, per_query=3,
        )

        yield _evt("retrieve", "done",
                   f"Selected {len(retrieved)} ICP snippets + "
                   f"{len(cust_retrieved)} customer snippets across "
                   f"{len({c.url for c in retrieved})} pages.",
                   data={"snippets": [c.text[:120] for c in retrieved[:6]],
                         "queries": _DEFAULT_QUERIES,
                         "customer_queries": _CUSTOMER_QUERIES})

        evidence_text, id_map = prompts.format_evidence(
            retrieved, max_chars=config.EVIDENCE_SNIPPET_CHARS
        )
        # Customer extractor gets its own formatted evidence (separate id_map)
        cust_evidence_text, cust_id_map = prompts.format_evidence(
            cust_retrieved, max_chars=config.EVIDENCE_SNIPPET_CHARS
        )

        # --- Find REAL named customers from MULTIPLE grounded sources (§0.6) --
        # A company's customers are visual + distributed (logo-wall images, press,
        # third-party articles), so site-scraping alone never generalises. We
        # merge three cited sources: (1) the sender's case-study links, (2) names
        # in the site text, (3) web search across the open web. Each customer
        # carries a source_url; nothing is invented.
        yield _evt("customers", "running",
                   "Finding the sender's customers (site links, page text, web search).")

        # Source 1+2: site — case-study link slugs + page text.
        # Enrich links with a brief snippet from each case-study page so the
        # synthesis can say "SumUp (payments fintech) bought for X reason."
        raw_links = [] if config.MOCK_MODE else find_customer_links(url)
        cust_links = ([] if config.MOCK_MODE
                      else enrich_customer_snippets(raw_links))
        candidates_block = "\n".join(
            f"- {l['hint']}  ({l['url']})"
            + (f"\n  Description: {l['snippet']}" if l.get('snippet') else "")
            for l in cust_links
        )
        # Customer extractor uses its own evidence pool (customer-targeted chunks)
        # so it can cite customer-page snippets while synthesis keeps clean E-ids.
        combined_cust_evidence = cust_evidence_text or evidence_text
        combined_cust_id_map   = cust_id_map if cust_retrieved else id_map
        site_raw = self.llm.json(
            prompts.CUSTOMERS_SYSTEM,
            prompts.customers_user(domain, combined_cust_evidence, candidates_block),
            task="sender_customers", max_tokens=500, model=config.MODEL_LIGHT,
        )
        link_by_hint = {l["hint"]: l for l in cust_links}
        site_customers: List[NamedCustomer] = []
        for c in site_raw.get("customers", []):
            name = (c.get("name") or "").strip()
            if not name:
                continue
            if _is_non_customer(name, c.get("source_url", "")):
                continue
            slug = (c.get("slug") or "").strip().lower()
            link_data = link_by_hint.get(slug, {})
            src = link_data.get("url", "")
            eids = [e for e in c.get("evidence_ids", []) if e in combined_cust_id_map]
            site_customers.append(NamedCustomer(
                name=name, evidence_ids=eids, source_url=src,
                provenance="case-study" if src else "site-text",
                # link_data snippet (live) or LLM-returned description (mock/fallback)
                description=link_data.get("snippet", "") or c.get("description", ""),
            ))
        # Safety net: any case-study slug the model dropped still becomes a customer.
        named_slugs = {(c.get("slug") or "").lower() for c in site_raw.get("customers", [])}
        for l in cust_links:
            if l["hint"] in named_slugs:
                continue
            slug_name = l["hint"].replace("-", " ").title()
            if _is_non_customer(slug_name, l.get("url", "")):
                continue
            site_customers.append(NamedCustomer(
                name=slug_name,
                source_url=l["url"], provenance="case-study",
                description=l.get("snippet", ""),
            ))

        # --- Step 2: web search for additional customers (runs BEFORE synthesis) -
        # Moved here so the ICP is derived from the COMPLETE customer list, not
        # just what the site scrape found. Web search often finds customers the
        # logo wall misses (press mentions, review sites, open-web case studies).
        web_customers: List[NamedCustomer] = []
        if config.CUSTOMER_WEB_SEARCH:
            yield _evt("customers", "running",
                       "Searching the web for additional customers…")
            web_customers = self._web_customers(domain)

        # Merge site + web, deduped by normalised name.
        named_customers: List[NamedCustomer] = _merge_customers(
            [site_customers, web_customers]
        )

        # --- Enrichment: industry + size for ALL customers -------------------
        # Runs on the merged list; works from description OR name alone for
        # well-known companies (model uses general knowledge when no description).
        to_enrich = [c for c in named_customers
                     if not (c.industry and c.size_hint)][:12]
        if to_enrich:
            def _enrich_line(c: NamedCustomer) -> str:
                if c.description:
                    return f"- {c.name}: {c.description}"
                if c.source_url:
                    # URL gives the model strong context for ambiguous names
                    # (e.g. "Remote" alone is unclear; "remote.com" is not)
                    return f"- {c.name} (see: {c.source_url})"
                return f"- {c.name}"
            pairs = "\n".join(_enrich_line(c) for c in to_enrich)
            try:
                enrich_raw = self.llm.json(
                    prompts.CUSTOMER_ENRICH_SYSTEM,
                    prompts.customer_enrich_user(pairs),
                    task="customer_enrichment",
                    max_tokens=500, model=config.MODEL_LIGHT,
                )
                lookup = {(r.get("name") or "").lower(): r
                          for r in enrich_raw.get("companies", [])}
                for c in named_customers:
                    data = lookup.get(c.name.lower())
                    if data:
                        c.industry  = data.get("industry",  "") or c.industry
                        c.size_hint = data.get("size_hint", "") or c.size_hint
            except Exception:  # noqa: BLE001 — enrichment is optional
                pass

        n_web = sum(1 for c in named_customers if c.provenance == "web")
        customers_block = "\n".join(
            f"- {c.name}"
            + (f" | industry: {c.industry}" if c.industry else "")
            + (f" | size: {c.size_hint}" if c.size_hint else "")
            + (f" | source: {c.source_url}" if c.source_url and not c.industry else "")
            for c in named_customers
        )
        yield _evt("customers", "done",
                   f"Found {len(named_customers)} customer(s) "
                   f"({len(named_customers) - n_web} from site, {n_web} from web).",
                   data={"named_customers": [c.model_dump() for c in named_customers]})

        # --- Corpus-level sufficiency check (before any LLM call) -------------
        total_chars = sum(len(c.text) for c in retrieved)
        corpus_thin = len(retrieved) < _THIN_CORPUS_CHUNKS or total_chars < _THIN_CORPUS_CHARS

        if corpus_thin:
            gaps = _detect_gaps(retrieved)
            yield _evt("synthesize", "running",
                       f"Corpus thin ({len(retrieved)} chunks, {total_chars} chars) — "
                       "using constrained prompt; marking gaps.")
            raw = self.llm.json(
                prompts.SENDER_SYSTEM,
                prompts.sender_user(domain, evidence_text, low_evidence=True,
                                    customers_block=customers_block),
                task="sender_synthesis",
                max_tokens=1600,
                model=config.MODEL,
            )
            profile = _build_profile(raw, domain, id_map,
                                     confidence="low",
                                     extra_gaps=gaps,
                                     evidence_sufficient=False,
                                     named_customers=named_customers)
        else:
            yield _evt("synthesize", "running",
                       "Asking Claude to infer the value proposition and ICP from "
                       "the retrieved snippets only.")
            raw = self.llm.json(
                prompts.SENDER_SYSTEM,
                prompts.sender_user(domain, evidence_text,
                                    customers_block=customers_block),
                task="sender_synthesis",
                max_tokens=1600,
                model=config.MODEL,
            )
            profile = _build_profile(raw, domain, id_map,
                                     named_customers=named_customers)

        # --- Adaptive re-plan: fires only when evidence is thin/low-confidence -
        # Skip if corpus was already too thin (replan won't help same corpus).
        adaptive_used = False
        if not corpus_thin and _is_low_confidence(profile, retrieved):
            adaptive_used = True
            yield _evt("replan", "running",
                       f"Evidence thin ({len(retrieved)} snippets) or low-confidence — "
                       "building tailored queries from site structure.")

            overview = _build_overview(pages)
            plan_raw = self.llm.json(
                prompts.PLANNER_SYSTEM,
                prompts.planner_user(overview),
                task="sender_planner",
                max_tokens=200,
                model=config.MODEL_LIGHT,
            )
            tailored = plan_raw.get("queries") or _DEFAULT_QUERIES
            retrieved = retriever.multi_search(tailored, per_query=3)
            evidence_text, id_map = prompts.format_evidence(retrieved)

            yield _evt("replan", "done",
                       f"Re-retrieved {len(retrieved)} snippets with "
                       f"{len(tailored)} tailored queries. Re-synthesizing…")

            raw = self.llm.json(
                prompts.SENDER_SYSTEM,
                prompts.sender_user(domain, evidence_text,
                                    customers_block=customers_block),
                task="sender_synthesis",
                max_tokens=1600,
                model=config.MODEL,
            )
            profile = _build_profile(raw, domain, id_map,
                                     named_customers=named_customers)

        # Guarantee: every named customer's industry appears in the ICP.
        # The LLM sometimes drops minority segments (e.g. "Event Tech" when most
        # customers are SaaS/Fintech). This makes it deterministic.
        if named_customers:
            icp_lower = [i.lower() for i in profile.icp.industries]
            for c in named_customers:
                if not c.industry:
                    continue
                c_low = c.industry.lower()
                if not any(c_low in ex or ex in c_low for ex in icp_lower):
                    profile.icp.industries.append(c.industry)
                    icp_lower.append(c_low)

        # Single synthesize-done event — always fires exactly once.
        cov_pct = int(profile.grounding_coverage * 100)
        if corpus_thin:
            done_detail = (f"Low-confidence profile — {len(profile.data_gaps)} gap(s) flagged. "
                           f"Grounding: {cov_pct}%.")
        elif adaptive_used:
            done_detail = f"Value proposition and ICP ready (adaptive path). Grounding: {cov_pct}%."
        else:
            done_detail = f"Value proposition and ICP ready. Grounding: {cov_pct}%."
        yield _evt("synthesize", "done", done_detail,
                   data={"profile": profile.model_dump(),
                         "adaptive": adaptive_used,
                         "usage": self.llm.meter.snapshot()})


    # ------------------------------------------------------------------ #
    # MODE 2 — target evaluation + outbound drafting
    # ------------------------------------------------------------------ #
    def run_target(self, url: str, persona_role: str, persona_seniority: str,
                   sender: SenderProfile, max_pages: int) -> Iterator[Dict]:
        persona = f"{persona_seniority} {persona_role}".strip()
        domain = url.replace("https://", "").replace("http://", "").strip("/")

        yield _evt("plan", "running",
                   f"Planning research on {url} for persona: {persona}.")

        _TARGET_COVERAGE_QUERIES = [
            "what the company does product industry",
            "customers who we serve market segment",
            "company size team employees scale",
            "funding round raised investment series",
            "hiring jobs careers open roles",
        ]
        yield _evt("fetch", "running", f"Fetching up to {max_pages} pages from {url}…")
        pages = fetch_site(url, max_pages=max_pages,
                           coverage_queries=_TARGET_COVERAGE_QUERIES)

        # If the site is sparse (JS-rendered or brand-new), pull in YC and
        # Product Hunt as supplementary evidence before scoring fit.
        total_chars = sum(len(p.text) for p in pages)
        if total_chars < config.ENRICH_THRESHOLD:
            enrich_pages = fetch_enrichment(domain)
            if enrich_pages:
                pages = pages + enrich_pages
                src_labels = ", ".join(p.url.split("/")[2] for p in enrich_pages)
                yield _evt("fetch", "done",
                           f"Fetched {len(pages) - len(enrich_pages)} page(s) from site "
                           f"(sparse: {total_chars} chars). Enriched from: {src_labels}.")
            else:
                yield _evt("fetch", "done",
                           f"Fetched {len(pages)} page(s) (sparse site; "
                           "no enrichment found on YC or Product Hunt).")
        else:
            yield _evt("fetch", "done", f"Fetched {len(pages)} pages.")

        # --- Web search: signal hunting from third-party sources ---------------
        # Always runs (not just on thin sites) because the target's own site
        # rarely reveals buying signals — press, Crunchbase, LinkedIn do.
        # This also solves sign-up-walled sites: third-party coverage is richer
        # for sales signals than the product pages we can't access anyway.
        yield _evt("research", "running",
                   f"Searching for recent signals on {domain} "
                   "(funding, hiring, launches, news)…")
        web_snippets: List[Page] = []
        if not config.MOCK_MODE:
            try:
                web_raw = self.llm.web_search_json(
                    prompts.TARGET_WEBSEARCH_SYSTEM,
                    prompts.target_websearch_user(domain),
                    task="target_web_research",
                    max_tokens=2500,
                    max_searches=6,
                    model=config.MODEL_LIGHT,
                    queries=prompts.target_websearch_queries(domain),
                )
                for s in web_raw.get("snippets") or []:
                    text = (s.get("text") or "").strip()
                    src  = (s.get("source_url") or domain).strip()
                    sig_type = (s.get("type") or "other").strip()
                    if text:
                        web_snippets.append(Page(
                            url=src,
                            title=f"[{sig_type}] {domain}",
                            text=text,
                        ))
            except Exception:  # noqa: BLE001 — web search is optional
                pass
        else:
            from . import mocks as _mocks
            mock_raw = _mocks.respond("target_web_research", domain)
            for s in mock_raw.get("snippets") or []:
                text = (s.get("text") or "").strip()
                if text:
                    web_snippets.append(Page(
                        url=(s.get("source_url") or domain),
                        title=f"[{s.get('type','other')}] {domain}",
                        text=text,
                    ))

        if web_snippets:
            pages = pages + web_snippets
            yield _evt("research", "done",
                       f"Found {len(web_snippets)} web signal(s) for {domain}.")
        else:
            yield _evt("research", "done",
                       f"No additional web signals found for {domain}.")

        chunks = chunk_pages(pages)
        retriever = Retriever(chunks)
        yield _evt("index", "done", f"Indexed {len(chunks)} passages (BM25).")

        # Retrieve for two purposes: signals (time-sensitive) and fit (firmographic).
        signal_q = [
            "funding round raised investment series",
            "hiring jobs careers we're hiring open roles",
            "new product launch announcement release",
            "expansion new office new market growth",
            "new ceo cto vp leadership appointment",
        ]
        fit_q = [
            "what the company does product industry",
            "customers who we serve market segment",
            "company size team employees scale",
        ]
        sig_chunks = retriever.multi_search(signal_q, per_query=1)
        fit_chunks = retriever.multi_search(fit_q, per_query=2)
        # Combined corpus for emails = union, de-duped by id.
        merged: Dict[str, Chunk] = {c.id: c for c in (sig_chunks + fit_chunks)}
        all_chunks = list(merged.values())
        # Apply overall evidence budget across the union before formatting.
        if config.EVIDENCE_BUDGET_CHARS:
            budget_chunks: List[Chunk] = []
            total_chars = 0
            for c in all_chunks:
                cost = min(len(c.text), config.EVIDENCE_SNIPPET_CHARS or len(c.text))
                if total_chars + cost <= config.EVIDENCE_BUDGET_CHARS:
                    budget_chunks.append(c)
                    total_chars += cost
            all_chunks = budget_chunks
        evidence_text, id_map = prompts.format_evidence(
            all_chunks, max_chars=config.EVIDENCE_SNIPPET_CHARS
        )
        yield _evt("retrieve", "done",
                   f"Selected {len(all_chunks)} snippets for signals + fit.")

        # --- signals + fit (one call — evidence_text sent once) --------------
        # Previously two calls (target_signals on haiku, target_fit on sonnet).
        # Now one sonnet call returns both; evidence is formatted and billed once.
        icp_json = json.dumps(sender.icp.model_dump(), indent=2)
        # Pass full customer profiles so Mode 2 can:
        # (a) detect direct name matches (RULE 0 override)
        # (b) recognise companies similar in type to known customers (RULE 0b boost)
        def _cust_line(c: NamedCustomer) -> str:
            parts = [c.name]
            if c.industry:
                parts.append(f"({c.industry})")
            if c.size_hint:
                parts.append(c.size_hint)
            return " — ".join(parts) if len(parts) > 1 else c.name

        known_lines = [_cust_line(c) for c in sender.named_customers if c.name]
        known_block = (
            "\n\nSENDER'S KNOWN CUSTOMERS (each entry = proven buyer):\n"
            + "\n".join(f"- {l}" for l in known_lines)
        ) if known_lines else ""

        yield _evt("signals", "running",
                   "Extracting signals and assessing fit — evidence sent once.")
        combined_raw = self.llm.json(
            prompts.SIGNALS_AND_FIT_SYSTEM,
            prompts.signals_and_fit_user(domain, evidence_text),
            task="target_signals_and_fit", max_tokens=1100,
            model=config.MODEL,
            # ICP + known customers cached — same for every target in this run.
            cache_prefix=f"SENDER ICP:\n{icp_json}{known_block}",
        )
        signals = [Signal(**s) for s in combined_raw.get("signals", [])]
        yield _evt("signals", "done", f"Found {len(signals)} signal(s).",
                   data={"signals": [s.model_dump() for s in signals]})

        # --- fit (dimensions returned by the combined call above) ----------
        yield _evt("fit", "running", "Computing fit score from returned dimension judgments.")

        # Parse and enforce grounding rules per dimension.
        dims_raw = combined_raw.get("dimensions", {})
        dimensions: Dict[str, DimensionJudgment] = {}
        for name in ("industry", "trigger", "size", "buyer", "geography"):
            d = dims_raw.get(name, {})
            judgment = d.get("judgment", "absent")
            eids = [e for e in d.get("evidence_ids", []) if e in id_map]
            if judgment == "absent":
                eids = []          # rule: absent → evidence_ids must be empty
            elif not eids:
                judgment = "absent"  # rule: non-absent without valid cites → downgrade
            dimensions[name] = DimensionJudgment(
                judgment=judgment, evidence_ids=eids, note=d.get("note", ""),
            )

        # Parse anti-pattern hits, keeping only those with valid evidence.
        anti_patterns_hit = [
            AntiPatternHit(
                name=ap.get("name", ""),
                evidence_ids=[e for e in ap.get("evidence_ids", []) if e in id_map],
            )
            for ap in combined_raw.get("anti_patterns_hit", [])
            if ap.get("name")
        ]

        # Compute deterministic score (transparency check only — not the headline).
        score, tier = score_fit(dimensions, anti_patterns_hit)

        # Parse qualification answers — one per ICP qualifying_question.
        qualification = [
            QuestionAnswer(
                question=(q.get("question") or "").strip(),
                answer=(q.get("answer") or "unknown").strip(),
                rationale=(q.get("rationale") or "").strip(),
                evidence_ids=[e for e in (q.get("evidence_ids") or []) if e in id_map],
            )
            for q in combined_raw.get("qualification", [])
            if (q.get("question") or "").strip()
        ]

        # LLM-judged headline verdict (pain fit + worth reaching out).
        raw_pain = combined_raw.get("pain_fit")
        pain_fit = max(0, min(100, int(raw_pain))) if raw_pain is not None else 0

        # Generate human-readable notes for backward-compat rendering.
        dimension_notes = [
            f"{name.title()}: {dim.judgment} — {dim.note}"
            for name, dim in dimensions.items()
            if dim.note or dim.judgment != "absent"
        ]

        # Union all cited ids into the evidence panel.
        all_fit_eids = [e for d in dimensions.values() for e in d.evidence_ids] + \
                       [e for ap in anti_patterns_hit for e in ap.evidence_ids] + \
                       [e for q in qualification for e in q.evidence_ids]
        fit_eids = list(dict.fromkeys(all_fit_eids))

        fit = FitScore(
            pain_fit=pain_fit,
            pain_fit_rationale=combined_raw.get("pain_fit_rationale", ""),
            worth_reaching_out=combined_raw.get("worth_reaching_out", ""),
            worth_rationale=combined_raw.get("worth_rationale", ""),
            outreach_angle=combined_raw.get("outreach_angle", ""),
            qualification=qualification,
            score=score,
            tier=tier,
            rationale=combined_raw.get("rationale", ""),
            dimensions=dimensions,
            anti_patterns_hit=anti_patterns_hit,
            dimension_notes=dimension_notes,
            evidence_ids=fit_eids,
        )
        cov = sum(1 for d in dimensions.values() if d.judgment != "absent")
        worth = fit.worth_reaching_out or fit.tier.lower()
        yield _evt("fit", "done", f"pain fit {fit.pain_fit} · {worth} ({cov}/5 dimensions)",
                   data={"fit": fit.model_dump()})

        # --- Strategist: pick strongest pain + trigger, brief both angles ----
        signals_text = "\n".join(
            f"- [{s.type}] {s.summary} (cites {', '.join(s.evidence_ids)})"
            for s in signals
        ) or "None detected."
        fit_summary = (
            f"pain_fit: {fit.pain_fit}/100. "
            f"Worth reaching out: {fit.worth_reaching_out}. "
            f"Outreach angle: {fit.outreach_angle}. "
            f"{fit.pain_fit_rationale}"
        )
        yield _evt("draft", "running", "Strategist choosing angles…")
        strategist_raw = self.llm.json(
            prompts.STRATEGIST_SYSTEM,
            prompts.strategist_user(domain, fit_summary, signals_text, evidence_text),
            task="target_strategist",
            max_tokens=400,
            model=config.MODEL_LIGHT,
        )
        angle_plan = (
            f"Pain angle — {strategist_raw.get('chosen_pain', '')}\n"
            f"{strategist_raw.get('pain_angle_brief', '')}\n\n"
            f"Trigger angle — {strategist_raw.get('chosen_trigger', '')}\n"
            f"{strategist_raw.get('trigger_angle_brief', '')}"
        )

        # --- Drafter: write two emails using the angle plan -----------------
        # Pass top named customers for social proof in the pain-led email.
        customers_block = "\n".join(
            f"- {c.name}" + (f" ({c.industry})" if c.industry else "")
            for c in sender.named_customers[:6] if c.name
        )
        yield _evt("draft", "running",
                   "Drafting two angle-differentiated emails from the strategy brief.")
        email_raw = self.llm.json(
            prompts.EMAIL_SYSTEM,
            prompts.email_user(persona, domain, signals_text, evidence_text,
                               angle_plan, customers_block),
            task="target_emails", max_tokens=900,
            model=config.MODEL,
            cache_prefix=f"SENDER value prop:\n{sender.value_proposition}",
        )

        def _parse_emails(raw_list: list) -> List[Email]:
            result = []
            for e in raw_list:
                claims = []
                for c in e.get("claims", []):
                    eids = c.get("evidence_ids", [])
                    grounded = bool(eids) and all(eid in id_map for eid in eids)
                    claims.append(Claim(text=c.get("text", ""),
                                        evidence_ids=eids, grounded=grounded))
                result.append(Email(
                    angle=e.get("angle", ""),
                    angle_label=e.get("angle_label", e.get("angle", "")),
                    subject=e.get("subject", ""),
                    body=e.get("body", ""),
                    claims=claims,
                ))
            return result

        emails: List[Email] = _parse_emails(email_raw.get("emails", []))

        # --- Critic: quality gate — reject generic lines, enforce rules ------
        yield _evt("draft", "running", "Critic reviewing emails…")
        emails_json = json.dumps([e.model_dump() for e in emails], indent=2)
        critic_raw = self.llm.json(
            prompts.CRITIC_SYSTEM,
            prompts.critic_user(emails_json, evidence_text),
            task="target_critic",
            max_tokens=1000,
            model=config.MODEL,
        )
        if not critic_raw.get("passed") and critic_raw.get("emails"):
            revised = _parse_emails(critic_raw["emails"])
            if revised:
                emails = revised

        # Build the evidence panel: union of every cited id across the report.
        cited_ids: List[str] = list(fit.evidence_ids)
        for s in signals:
            cited_ids += s.evidence_ids
        for e in emails:
            for c in e.claims:
                cited_ids += c.evidence_ids
        evidence_panel = _evidence_from_ids(cited_ids, id_map)

        report = TargetReport(
            target_company=domain,
            persona_role=persona_role,
            persona_seniority=persona_seniority,
            summary=combined_raw.get("summary", ""),
            signals=signals,
            fit=fit,
            emails=emails,
            evidence=evidence_panel,
        )
        yield _evt("draft", "done", "Emails and claim map ready.",
                   data={"report": report.model_dump(),
                         "usage": self.llm.meter.snapshot()})
