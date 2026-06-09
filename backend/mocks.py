"""Realistic mock responses for MOCK_MODE.

Two rules:
  1. Every response that cites an evidence id must cite a real E-id that
     exists in the prompt — _eids() extracts them from the user string.
  2. target_signals and target_fit are DERIVED from the retrieved evidence
     text, not from the company name. The same prompt in → the same response
     out; a different prompt in → a different response out.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _eids(user: str, n: int = 3) -> List[str]:
    """Return up to n unique E-ids that appear in the prompt."""
    ids = re.findall(r"\bE\d+\b", user)
    seen: List[str] = []
    for i in ids:
        if i not in seen:
            seen.append(i)
    return seen[:n] or ["E1"]


def _evidence_chunks(user: str) -> List[Tuple[str, str]]:
    """Parse [E1]…[E2]… blocks → [(eid, lowercased_body), …].

    Works for both signals_user and fit_user prompts because both embed
    the evidence text inline with [E#] markers.
    """
    parts = re.split(r"(\[E\d+\])", user)
    chunks: List[Tuple[str, str]] = []
    i = 0
    while i < len(parts):
        if re.match(r"^\[E\d+\]$", parts[i]):
            eid  = parts[i][1:-1]                            # strip brackets
            body = parts[i + 1].lower() if i + 1 < len(parts) else ""
            chunks.append((eid, body))
            i += 2
        else:
            i += 1
    return chunks


def _find_eid_with_kw(
    chunks: List[Tuple[str, str]],
    keywords: List[str],
) -> Tuple[str, str]:
    """Return (first_eid, matched_keyword) for any chunk containing a keyword.

    Returns ("", "") when no keyword is found in any chunk.
    """
    for eid, text in chunks:
        for kw in keywords:
            if kw in text:
                return eid, kw
    return "", ""


def _assess(
    chunks: List[Tuple[str, str]],
    present_kws: List[str],
    partial_kws: List[str] = (),
    absent_kws:  List[str] = (),
) -> Tuple[str, str, str]:
    """Return (judgment, eid, note) by scanning evidence chunks.

    absent_kws are checked across ALL chunks first — if any chunk contains a
    disqualifying keyword the dimension is absent regardless of other content.
    present_kws are checked next; partial_kws last.
    """
    if absent_kws:
        for eid, text in chunks:
            for kw in absent_kws:
                if kw in text:
                    return "absent", "", f"disqualifying pattern '{kw}' in evidence"
    eid, kw = _find_eid_with_kw(chunks, present_kws)
    if eid:
        return "present", eid, f"'{kw}' in evidence"
    if partial_kws:
        eid, kw = _find_eid_with_kw(chunks, list(partial_kws))
        if eid:
            return "partial", eid, f"'{kw}' in evidence"
    return "absent", "", "no matching keywords in evidence"


# ---------------------------------------------------------------------------
# Mock dispatch
# ---------------------------------------------------------------------------

def respond(task: str, user: str) -> Dict[str, Any]:

    # ── Mode 1: sender ICP synthesis ────────────────────────────────────────
    if task == "sender_synthesis":
        e = _eids(user, 5)
        return {
            "one_liner": "An AI BDR that runs outbound end-to-end.",
            "customer_pattern": "Most customers are B2B SaaS and fintech startups (1-500 employees) with active outbound sales teams, at Series A-C stage, buying to scale pipeline without hiring more SDRs.",
            "value_proposition": (
                "Replaces the cost and overhead of a human outbound team with an "
                "autonomous AI BDR that sources leads, writes personalized multi-channel "
                "outreach, handles replies, and books meetings — generating pipeline at a "
                "fraction of the cost of hiring."
            ),
            "capabilities": [
                "Sources leads from a 250M+ verified B2B contact database",
                "Monitors intent signals like funding rounds and leadership hires",
                "Writes hyper-personalized email and social sequences",
                "Handles replies, objections, and books meetings autonomously",
            ],
            "icp": {
                "industries": ["B2B SaaS", "Fintech", "Marketplaces", "Tech-enabled services"],
                "size_bands": [
                    {"label": "Startups (1-50)",        "rationale": "Need pipeline before hiring a sales team."},
                    {"label": "SMB / Mid-market (50-1000)", "rationale": "Scaling outbound without scaling headcount."},
                ],
                "triggers": [
                    {"name": "Recent funding round",  "description": "New capital to deploy on growth/GTM.",
                     "evidence_ids": e[:1]},
                    {"name": "Sales leadership hire", "description": "New VP Sales / CRO building a motion.",
                     "evidence_ids": e[1:2] or e[:1]},
                    {"name": "Hiring BDRs/SDRs",      "description": "Outbound pain is acute and budgeted.",
                     "evidence_ids": e[2:3] or e[:1]},
                ],
                "buyers": [
                    {"role": "Head of Sales / VP Sales", "seniority": "VP / Director",
                     "why": "Owns pipeline targets.",    "evidence_ids": e[:1]},
                    {"role": "Founder / CEO",             "seniority": "C-level",
                     "why": "Early-stage outbound owner.", "evidence_ids": e[1:2] or e[:1]},
                    {"role": "Head of Growth / RevOps",   "seniority": "Director",
                     "why": "Owns GTM efficiency.",       "evidence_ids": e[2:3] or e[:1]},
                ],
                "geographies": ["North America", "Europe"],
                "anti_patterns": [
                    "Pure PLG companies with no outbound motion",
                    "Enterprises requiring heavy procurement and on-prem",
                ],
                "qualifying_questions": [
                    "Does the company run an outbound sales motion (BDRs/SDRs/AEs)?",
                    "Are they 1-1000 employees (startup to mid-market)?",
                    "Is there a recent trigger (funding, sales-leadership hire, BDR hiring)?",
                    "Is a VP Sales / Head of Growth / founder the likely buyer?",
                ],
            },
            "evidence_ids": _eids(user, 5),
        }

    # ── Mode 1: named customer extraction (§0.6) ────────────────────────────
    if task == "sender_customers":
        # MOCK_MODE has no live case-study links, so return a small grounded
        # sample from the evidence so the path is exercised offline.
        e = _eids(user, 3)
        return {"customers": [
            {"name": "SumUp", "slug": "", "evidence_ids": e[:1],
             "description": "Global payments and financial services platform for SMBs."},
            {"name": "Clay",  "slug": "", "evidence_ids": e[1:2] or e[:1],
             "description": "Data enrichment and outbound automation tool for sales teams."},
        ]}

    # ── Mode 1: bounded enrichment — industry + size per customer ────────────
    if task == "customer_enrichment":
        return {"companies": [
            {"name": "SumUp",          "industry": "Fintech",      "size_hint": "Series E"},
            {"name": "Clay",           "industry": "B2B SaaS",     "size_hint": "Series B"},
            {"name": "Quora",          "industry": "B2C Platform", "size_hint": "enterprise"},
            {"name": "SaaStr",         "industry": "Events",       "size_hint": "bootstrapped"},
            {"name": "CookUnity",      "industry": "Food Tech",    "size_hint": "Series C"},
            {"name": "Chain of Events","industry": "Events",       "size_hint": "startup"},
        ]}

    # ── Mode 1: web-search customer discovery (open web, not just the site) ──
    if task == "sender_customers_web":
        # MOCK has no live web; return a representative cited set incl. one the
        # site alone would miss (Quora), so the offline demo shows the source.
        return {"customers": [
            {"name": "Quora",     "source_url": "https://www.ycombinator.com/companies/artisan"},
            {"name": "SaaStr",    "source_url": "https://www.artisan.co/customers/saastr"},
            {"name": "CookUnity", "source_url": "https://www.artisan.co/customers/cookunity-case-study"},
        ]}

    # ── Mode 2: target web research (signal hunting from third-party sources) ──
    if task == "target_web_research":
        return {"snippets": [
            {"type": "funding",
             "text": "Recently closed a Series B funding round to expand its go-to-market team.",
             "source_url": "https://mock-web-research/crunchbase"},
            {"type": "hiring",
             "text": "Actively hiring Account Executives and Sales Development Representatives.",
             "source_url": "https://mock-web-research/linkedin"},
            {"type": "overview",
             "text": "B2B SaaS platform serving mid-market companies with 50-500 employees.",
             "source_url": "https://mock-web-research/techcrunch"},
        ]}

    # ── Mode 2: combined signals + fit (one call, evidence sent once) ───────
    if task == "target_signals_and_fit":
        chunks = _evidence_chunks(user)

        # Signals — same keyword scan as the old target_signals mock.
        signals = []
        for sig_type, kws, summary_prefix in [
            ("funding",    ["raised", "series", "funding round", "investment round", "venture"],
                           "Funding signal"),
            ("hiring",     ["hiring", "open roles", "bdr", "account executive",
                            "sales development", "we're hiring", "open positions"],
                           "Hiring signal"),
            ("expansion",  ["expanding", "expansion", "new market", "new office", "launched in"],
                           "Expansion signal"),
            ("leadership", ["appointed", "joins as", "new ceo", "new cto", "new vp",
                            "chief revenue officer", "leadership hire"],
                           "Leadership change"),
            ("launch",     ["launched", "announcing", "new product", "new feature", "release"],
                           "Product launch"),
        ]:
            eid, kw = _find_eid_with_kw(chunks, kws)
            if eid:
                signals.append({
                    "type":         sig_type,
                    "summary":      f"{summary_prefix}: '{kw}' found in evidence.",
                    "recency":      "recent",
                    "evidence_ids": [eid],
                })

        # Fit dimensions — same keyword scan as the old target_fit mock.
        ind_j, ind_eid, ind_note = _assess(
            chunks,
            present_kws=["b2b saas", "b2b software", "fintech", "saas platform",
                         "enterprise software"],
            partial_kws=["b2b", "software", "platform", "marketplace", "technology company"],
        )
        trig_j, trig_eid, trig_note = _assess(
            chunks,
            present_kws=["raised", "series", "funding round", "hiring bdr",
                         "hiring sales development", "venture round"],
            partial_kws=["hiring", "expanding", "new product launch", "investment"],
        )
        buy_j, buy_eid, buy_note = _assess(
            chunks,
            present_kws=["vp of sales", "head of sales", "vp sales", "chief revenue",
                         "account executive", "bdr", "sales development representative"],
            partial_kws=["sales team", "revenue team", "growth team"],
        )
        size_j, size_eid, size_note = _assess(
            chunks,
            present_kws=["startup", "early stage", "100 employees", "200 employees",
                         "50 employees", "seed round", "series a", "series b"],
            partial_kws=["mid-market", "growing company", "employees", "team of", "headcount"],
            absent_kws= ["30,000 employees", "50,000 employees", "100,000 employees",
                         "fortune 500", "public company"],
        )
        geo_j, geo_eid, geo_note = _assess(
            chunks,
            present_kws=["north america", "united states", "us-based", "san francisco",
                         "new york", "europe", "london", "singapore", "global"],
            partial_kws=["international", "worldwide", "multiple countries"],
        )
        aps = []
        plg_eid, plg_kw = _find_eid_with_kw(
            chunks, ["open source", "self-serve", "developer-led", "plg", "product-led"]
        )
        if plg_eid:
            aps.append({"name": f"PLG/self-serve motion ('{plg_kw}' in evidence)",
                        "evidence_ids": [plg_eid]})
        big_eid, big_kw = _find_eid_with_kw(
            chunks, ["fortune 500", "30,000 employees", "50,000 employees",
                     "enterprise procurement"]
        )
        if big_eid:
            aps.append({"name": f"Large-enterprise scale ('{big_kw}' in evidence)",
                        "evidence_ids": [big_eid]})

        def _dim(j, eid, note):
            return {"judgment": j, "evidence_ids": [eid] if eid else [], "note": note}

        present_dims = [k for k, v in [("industry", ind_j), ("trigger", trig_j),
                                        ("buyer", buy_j), ("size", size_j),
                                        ("geography", geo_j)] if v == "present"]
        absent_dims  = [k for k, v in [("industry", ind_j), ("trigger", trig_j),
                                        ("buyer", buy_j), ("size", size_j),
                                        ("geography", geo_j)] if v == "absent"]
        parts = []
        if present_dims: parts.append(f"Evidence supports: {', '.join(present_dims)}.")
        if absent_dims:  parts.append(f"No evidence for: {', '.join(absent_dims)}.")
        if aps:          parts.append("Anti-pattern(s) detected.")

        # Pain fit: holistic score derived from how many dimensions have evidence.
        evidenced = sum(1 for j in [ind_j, trig_j, buy_j, size_j, geo_j] if j != "absent")
        pain_fit_score = min(90, max(10, evidenced * 18))
        worth = "yes" if pain_fit_score >= 60 else "maybe" if pain_fit_score >= 35 else "no"

        # Qualification: produce one answer per question found in the user prompt.
        # In mock mode we don't have the ICP questions, so produce a representative set.
        qualification = [
            {"question": "Does the company run an outbound sales motion (BDRs/SDRs/AEs)?",
             "answer": "yes" if buy_j != "absent" else "unknown",
             "rationale": "Buyer evidence found in retrieved snippets." if buy_j != "absent" else "No buyer evidence in retrieved content.",
             "evidence_ids": [buy_eid] if buy_eid else []},
            {"question": "Are they 1-1000 employees (startup to mid-market)?",
             "answer": "yes" if size_j == "present" else "partial" if size_j == "partial" else "unknown",
             "rationale": "Size evidence found." if size_j != "absent" else "No headcount information found.",
             "evidence_ids": [size_eid] if size_eid else []},
            {"question": "Is there a recent trigger (funding, sales-leadership hire, BDR hiring)?",
             "answer": "yes" if trig_j == "present" else "partial" if trig_j == "partial" else "no",
             "rationale": "Trigger signal detected in evidence." if trig_j != "absent" else "No trigger detected.",
             "evidence_ids": [trig_eid] if trig_eid else []},
        ]

        return {
            "signals": signals,
            "dimensions": {
                "industry":  _dim(ind_j,  ind_eid,  ind_note),
                "trigger":   _dim(trig_j, trig_eid, trig_note),
                "buyer":     _dim(buy_j,  buy_eid,  buy_note),
                "size":      _dim(size_j, size_eid, size_note),
                "geography": _dim(geo_j,  geo_eid,  geo_note),
            },
            "anti_patterns_hit": aps,
            "rationale": " ".join(parts) or "No matching evidence found.",
            "qualification": qualification,
            "pain_fit": pain_fit_score,
            "pain_fit_rationale": (
                f"Target shows {evidenced}/5 evidenced ICP dimensions. "
                + (" Matches core industry and buyer pattern." if evidenced >= 3
                   else " Partial match — limited evidence; inferred from company profile." if evidenced >= 1
                   else " Insufficient evidence to assess fit; recommend manual review.")
            ),
            "worth_reaching_out": worth,
            "worth_rationale": (
                "Strong dimension match and trigger present — good timing."
                if worth == "yes" else
                "Partial match; worth a light-touch outreach to validate fit."
                if worth == "maybe" else
                "Too few matching dimensions to justify outreach at this time."
            ),
            "outreach_angle": (
                "Lead with the recent trigger and connect it to pipeline efficiency."
                if worth == "yes" else
                "Explore whether they have an outbound motion being built; offer a pilot."
                if worth == "maybe" else
                "None — no evidence of an outbound sales motion or budget for sales tooling."
            ),
        }

    # ── Mode 2: strategist — picks angles before drafting ───────────────────
    if task == "target_strategist":
        e = _eids(user, 3)
        e1 = e[0] if e else "E1"
        e2 = e[1] if len(e) > 1 else e1
        return {
            "chosen_pain": "Scaling outbound pipeline without growing the SDR headcount",
            "chosen_trigger": f"Recent hiring or funding signal detected in evidence ({e1})",
            "pain_angle_brief": (
                f"Open with the cost and slowness of hiring SDRs to hit pipeline targets. "
                f"The persona owns revenue goals and knows each new hire takes months to ramp. "
                f"Connect to Artisan replacing that overhead with an AI BDR (cite {e1})."
            ),
            "trigger_angle_brief": (
                f"Open by referencing the detected signal ({e2}). "
                f"New capital or growth means immediate pressure to show pipeline ROI. "
                f"Position Artisan as the fastest way to stand up outbound without a hiring cycle."
            ),
        }

    # ── Mode 2: critic — quality gate on drafted emails ──────────────────────
    if task == "target_critic":
        # In mock mode always pass — the drafter mock already produces clean emails.
        return {"passed": True, "issues": [], "emails": []}

    # ── Mode 2: email drafting ───────────────────────────────────────────────
    if task == "target_emails":
        e = _eids(user, 4)
        e1 = e[0] if e else "E1"
        e2 = e[1] if len(e) > 1 else e1
        e3 = e[2] if len(e) > 2 else e1
        return {"emails": [
            {
                "angle": "pain-led", "angle_label": "Pain-led",
                "subject": "scaling pipeline without scaling the team",
                "body": (
                    "Hi {first_name},\n\n"
                    "Saw you're building out the go-to-market team — usually that means "
                    "outbound is about to get expensive fast: more reps, more tools, and "
                    "still no predictable pipeline.\n\n"
                    "We give teams an AI BDR that sources leads, writes personalized "
                    "outreach, and books meetings — pipeline at a fraction of the cost of "
                    "another hire.\n\nWorth a 15-min look?\n\nBest,\nAva"
                ),
                "claims": [
                    {"text": "The target is building out its go-to-market team.",
                     "evidence_ids": [e2]},
                    {"text": "Our product sources leads, writes outreach, and books meetings.",
                     "evidence_ids": [e1]},
                ],
            },
            {
                "angle": "trigger-led", "angle_label": "Trigger-led",
                "subject": "congrats on the raise — a thought on deploying it",
                "body": (
                    "Hi {first_name},\n\n"
                    "Congrats on the recent raise. The fastest place new capital usually "
                    "needs to show return is pipeline.\n\n"
                    "Teams in your spot use our AI BDR to stand up outbound in days, not "
                    "quarters.\n\nOpen to a quick look?\n\nBest,\nAva"
                ),
                "claims": [
                    {"text": "The target recently raised funding.",
                     "evidence_ids": [e1]},
                    {"text": "Our AI BDR can stand up outbound quickly.",
                     "evidence_ids": [e3]},
                ],
            },
        ]}

    # ── Mode 1: sender web research (thin-site enrichment) ──────────────────
    if task == "sender_web_research":
        return {
            "summary": (
                "A B2B SaaS platform that automates outbound sales. "
                "Serves sales teams at Series A–C startups and mid-market companies. "
                "Recently expanded with new AI-driven personalization features."
            ),
            "snippets": [
                {"text": "Helps BDR and SDR teams scale outbound without adding headcount.",
                 "source_url": "https://mock-web-research/g2-review"},
                {"text": "Series A backed; 50-200 employees; focused on B2B SaaS and fintech verticals.",
                 "source_url": "https://mock-web-research/crunchbase"},
            ],
        }

    # ── Mode 1: adaptive planner ─────────────────────────────────────────────
    if task == "sender_planner":
        return {"queries": [
            "what we do product platform",
            "customers industries we serve",
            "pricing plans segments",
            "outcomes results testimonials",
            "buyers decision makers roles",
        ]}

    if task == "target_summary":
        return {"summary": (
            "A venture-backed B2B software company serving mid-market customers, "
            "currently expanding its go-to-market team."
        )}

    return {}
