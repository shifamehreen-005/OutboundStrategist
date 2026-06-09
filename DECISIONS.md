# Engineering Decisions

This document records every material design and architectural decision made during development — what was built, why, and what was deliberately left out. It is written for a technical reviewer who wants to understand the reasoning, not just the outcome.

---

## 1. What it is

An agentic outbound tool that converts a sender's public website into a grounded Ideal Customer Profile, then evaluates any target account against that ICP and produces two cited outbound emails. Every factual claim in the output traces to a retrieved evidence snippet. No claim may reach the user without either a citation or an explicit unverified flag.

---

## 2. Core thesis

The tool is built directly against two documented weaknesses of existing AI BDR products:

**Templated output.** Generic "hyper-personalisation" that references only the company name and job title. Fixed by requiring every email claim to cite a retrieved evidence id. A claim without a valid citation is flagged `grounded=false` and rendered visibly in the UI. Fabricated personalisation is a visible failure, not a silent one.

**Single-agent shallowness.** One prompt that does everything produces shallow, averaged output. Fixed by a specialised multi-agent chain: each agent has one job, its own system prompt, a typed output contract, and the cheapest model capable of that job. The Strategist briefs the angles; the Drafter writes to the brief; the Critic enforces quality rules. The result is structurally different emails rather than two rewordings of the same message.

---

## 3. Pipeline architecture

### 3.1 Mode 1 — Sender → ICP

The pipeline is:

1. **Fetch** — crawl the sender's site (up to 4 prioritised pages: product, pricing, customers, about). Playwright JS-render fallback fires when static text is below 600 characters.

2. **Thin-site enrichment** — if the total crawled text is below 2,000 characters, two supplementary sources are pulled before indexing: YC and Product Hunt static pages (via `fetch_enrichment`), and a 3-query web search for general positioning, funding, and review information. This prevents an impoverished ICP when the sender's own site is sparse.

3. **Chunk + BM25** — passages (~700 chars, overlapping, heading-aware) are indexed in a BM25Okapi store. Five ICP-focused queries and three customer-focused queries retrieve the top evidence.

4. **Customer discovery — before ICP synthesis.** Real named customers are found from three sources in sequence: (a) case-study URL slugs scraped from the site, (b) company names in page text, (c) live web search across press, review sites, and third-party articles. All three are merged and deduplicated. The merged list is then enriched with industry and size tags via a single light-model batch call. Source URLs are passed as context for ambiguous names (e.g. `remote.com` disambiguates "Remote"). This step runs *before* synthesis so the ICP is derived from the complete, enriched customer list, not a partial one.

5. **ICP synthesis** — the main model reads the retrieved evidence and the complete customer list, and produces the value proposition, ICP dimensions, and 3–5 qualifying questions. The system prompt explicitly instructs the model to derive industries from the actual named buyers — classified by their buying unit, not their end product — rather than from the sender's aspirational marketing copy.

6. **Adaptive re-plan** — if the first synthesis is low-confidence (fewer than 4 retrieved chunks, value proposition under 60 characters, or no industries found), the light model generates 5–7 tailored BM25 queries from a compact site overview and the evidence is re-retrieved. On rich sites this step never fires, incurring zero additional cost.

### 3.2 Mode 2 — Target → fit + outbound

The pipeline is:

1. **Fetch** — crawl the target site. HTTP 403/429 errors on the root page are caught and treated as a non-fatal sparse-site condition rather than a hard failure. Web search then covers what the site cannot provide.

2. **Signal Hunter** — six web search queries always run, regardless of site richness: funding, hiring, news, leadership, company overview, and a broad open search on the company name. Third-party sources (Crunchbase, LinkedIn, TechCrunch, Wikipedia, press) surface buying signals and company background that the target's own site rarely exposes. The web snippets are added to the corpus before BM25 indexing, so they feed both signal detection and fit scoring.

3. **BM25 retrieve** — signal queries and fit queries are run over the merged corpus (site pages + web snippets). Results are union-merged and budget-capped before formatting.

4. **Signals + Fit Scorer** — a single main-model call receives the evidence once and returns: buying signals (each citing ≥1 evidence id); per-dimension judgments (industry, trigger, buyer, size, geography — strict: present/partial must cite ≥1 id, absent must have empty ids); qualifying question answers (yes/partial/no/unknown, each with rationale and optional citation); pain fit (0–100, holistic LLM judgment); worth reaching out (yes/maybe/no); and outreach angle. The ICP JSON and known customers are passed as a cached prefix and billed at 0.1× input cost on the second and subsequent target calls within the same session.

5. **Strategist** — the light model receives the fit summary and signals and picks the single strongest pain and the single strongest trigger. It writes a 2–3 sentence brief for each email angle, giving the Drafter a concrete starting point rather than a blank canvas. This is the mechanism that enforces structural differentiation: the pain-led and trigger-led emails are briefed differently from the outset.

6. **Drafter** — the main model writes both emails to the angle briefs. House rules are embedded in the system prompt: body under 125 words; paragraphs of at most 2 sentences; subject line 4 words or fewer, all lowercase; no em dashes; no spam words; no banned openers; outcomes not features; social proof from the sender's known customers; measurable metric in the pain-led email; process-question CTA (not a meeting request) in the pain-led email; trigger-led email must open with the named event.

7. **Critic** — the main model runs a 16-point checklist against both drafted emails. Any violation causes the offending email to be rewritten. If all rules pass, the originals are returned untouched. The Critic is the enforcement layer; the Drafter is the generation layer.

---

## 4. Key design decisions

### 4.1 LLM-judged fit score as the headline verdict

**Decision.** The primary fit verdict is the LLM's holistic pain fit score (0–100) and worth-reaching-out judgment. The deterministic Python formula (`scoring.py`) is retained but demoted to a transparency check displayed in a collapsible secondary section.

**Rationale.** A good fit decision is a judgment, not arithmetic. The question "do we solve a problem this company actually has, and are they worth a rep's time?" cannot be reliably answered by summing weighted dimension flags. The LLM can reason about hidden use cases, adjacent teams, and non-obvious angles that a rule-based formula cannot. For example, an event company may have a sponsor acquisition team that benefits from outbound automation even though none of the five ICP dimensions match.

The deterministic formula remains because it is reproducible, auditable, and company-name-blind — properties the LLM score cannot guarantee. Showing both communicates that the verdict is a judgment, not a calculation, while the calculation is available as a sanity check.

**Anti-fabrication guard preserved.** Dimension judgments retain the strict cite-or-absent rule: present or partial must cite at least one evidence id; absent must have empty ids. This rule is enforced in code after the LLM call and cannot be overridden by the model's output.

### 4.2 Customer discovery before ICP synthesis

**Decision.** Web search for additional customers was moved from after synthesis to before it, merged with site-scraped customers, and enriched with industry and size tags before the ICP synthesis call receives the list.

**Rationale.** When web search ran after synthesis, the ICP was derived from an incomplete customer list. A customer found only via web search (e.g. Quora, Remote) had no influence on the ICP industries, qualifying questions, or customer pattern. Moving discovery before synthesis means the ICP reflects the complete set of known buyers.

**Industry derivation from actual buyers.** The synthesis system prompt explicitly instructs the model to derive ICP industries from the named customers' buying units — not from the sender's marketing copy. A consumer tech company with a B2B advertising sales team is classified by its buying unit ("consumer tech / platforms with B2B sales teams"), not its end product ("consumer Q&A"). Marketing copy industries are overridden by the real customer pattern.

### 4.3 Known customers passed to Mode 2

**Decision.** The sender's named customers (with industry and size) are included in the Mode 2 cache prefix alongside the ICP JSON.

**Rationale.** A company that is already a customer of the sender is a proven buyer. Without this context, the fit scorer would assess it the same as any unknown target — often scoring it low because its public website does not expose the internal sales team that motivated the original purchase. With the customer list in context, the model can reason about re-engagement angles and recognise similar companies (e.g. Reddit when Quora is a known customer) and weight them accordingly.

The model reasons through this rather than applying a mechanical override. The prompt frames known customers as strong evidence of fit, not as an instruction to ignore contradicting signals.

### 4.4 Web search for Mode 2 targets — always, not just when sparse

**Decision.** The Signal Hunter web search runs for every Mode 2 target, regardless of whether the target's site returned sufficient content. Six queries run: funding, hiring, news, leadership, company overview, and a broad open search on the company name.

**Rationale.** A company's own website rarely exposes the signals that determine sales timing — funding rounds, active hiring in sales roles, leadership changes, product launches. These are reported by Crunchbase, TechCrunch, LinkedIn, and press, not on the homepage. Running web search only for sparse sites would miss these signals for all large companies with well-maintained public sites. The broad background query ensures that even JS-heavy or sparse sites (e.g. Revolut) produce citable evidence — without it, Claude falls back to world knowledge and cannot cite sources. Sites that actively block scraping (e.g. 403 responses) are also handled gracefully: the site fetch continues with whatever it could retrieve and the web search compensates for the gap.

### 4.11 Tavily as the primary web search backend

**Decision.** When `TAVILY_API_KEY` is set, all three web-search calls in the pipeline (Mode 1 customer discovery, Mode 1 thin-site enrichment, Mode 2 Signal Hunter) route through Tavily's REST API instead of Claude's native web_search tool.

**Rationale.** Claude's native web search had two reliability problems. First, the model generates verbose `<cite>` tags around every fact, pushing output past `max_tokens` for information-dense companies (e.g. Revolut). The JSON was then truncated mid-object and unparseable — a silent failure that produced "no evidence retrieved" in the UI. Second, native search is billed as output tokens on the LLM call, making it more expensive than a dedicated search API.

Tavily fixes both: it returns structured results (title/url/content) via a simple POST, so the LLM only needs to extract JSON from results it is given — not generate the search results itself. The six Mode 2 queries run concurrently via `ThreadPoolExecutor`, making the search portion ~6× faster than sequential. A truncation-salvage helper (`_salvage_array_objects`) was also added to `llm.py` as a safety net for the native fallback path.

The native path is kept as a fallback so the pipeline works for anyone who only has an Anthropic key, consistent with the MOCK_MODE / graceful-degradation principle.

### 4.13 Citation enforcement — evidence over world knowledge

**Decision.** The `pain_fit_rationale` and `worth_rationale` fields in the signals-and-fit prompt are required to cite evidence E-ids for every factual claim about the target. World knowledge may not be used silently.

**Rationale.** Without this rule, Claude writes accurate but uncitable analysis from training data (e.g. "Revolut Business operates across 35+ markets"). The analysis looks correct but the evidence panel is empty — fabricated personalisation is a silent failure, not a visible one. The citation rule forces Claude to ground every claim in a fetched snippet, making any gap visible rather than hidden. When evidence is genuinely absent, the claim is omitted rather than invented.

### 4.5 Strategist + Critic as separate agents

**Decision.** The email generation chain was split into three agents: Strategist (light model), Drafter (main model), Critic (main model). The research, signals, and fit scoring stages remain a single combined call.

**Rationale.** The email quality problem — generic, structurally identical drafts — is not primarily a model capability problem. It is a briefing problem: the Drafter was being asked to simultaneously decide what to write about and how to write it. The Strategist removes the first decision by specifying exactly which pain and which trigger to use and how to frame each angle. The Drafter then executes to a brief rather than making choices under a broad instruction.

The Critic addresses a different failure mode: rules that the Drafter knows but may violate under generation pressure (token limits, instruction conflicts). An independent review pass with an explicit checklist catches violations before they reach the user.

Keeping research, signals, and fit as a single call is deliberate. Splitting them would triple the billable calls for no quality improvement — the signal and fit evidence is the same corpus, formatted once. The token discipline mandate from `CLAUDE.md` applies.

### 4.6 Email quality rules

The following rules are embedded in both the Drafter prompt and the Critic checklist, and are enforced at two independent layers:

- Body under 125 words
- Paragraphs of at most 2 sentences
- Subject line: 4 words or fewer, all lowercase
- No em dashes
- No spam words (guarantee, free, revolutionary, game-changing)
- No banned openers (I wanted to reach out, hope this finds you, My name is, I came across)
- No generic praise (love your work, amazing, impressive)
- No feature listing — outcomes only
- Pain-led: one specific workflow bottleneck, named negative outcome, social proof from a known customer, measurable metric, process-question CTA (not a meeting or demo request)
- Trigger-led: opens with the named event
- Every company-specific claim cites at least one evidence id
- The two emails are structurally different

### 4.7 Outreach angle for every target

**Decision.** The fit scorer returns an `outreach_angle` field for every target, including low-fit ones. If no realistic angle exists, the field begins with "None —" followed by a one-line reason.

**Rationale.** A fit score alone does not tell a sales representative what to do. A score of 20 could mean "no realistic case" or "possible angle via an adjacent team not visible on the public site." The outreach angle surfaces the model's judgment about what a rep could actually say, which is more actionable than a number. Requiring a reason for "none" prevents the field from being a silent disqualification.

### 4.8 BM25 retrieval over vector embeddings

**Rationale.** At inference time the corpus is 15–30 chunks from one or two sites. Lexical BM25 gives strong, explainable recall at this scale with zero embedding cost and no second API dependency. The queries are hand-crafted for the ICP dimensions being retrieved, which makes lexical matching effective. Replacing the retriever with a hybrid approach (BM25 + embedding reranker) is a one-file change to `retriever.py` if query-term mismatch becomes a recurring problem.

### 4.9 MOCK_MODE

**Rationale.** The application must run end-to-end without an API key. This is a hard constraint (`CLAUDE.md`). Every LLM call has a corresponding mock branch in `mocks.py`. The mock for `target_signals_and_fit` scans the actual evidence text in the prompt for keywords — it reacts to retrieved content, not the company name — so mock runs exercise the same evidence-to-judgment path as live runs, just with a regex engine instead of Claude. Every new agent call added during development received a mock branch before the code was committed.

### 4.10 Prompt caching

The ICP JSON (passed as `cache_prefix` on the `target_signals_and_fit` call) and the sender value proposition (passed as `cache_prefix` on the email call) are cached at the Anthropic API layer with a 5-minute TTL. On the second and subsequent Mode 2 target calls within the same session, these prefixes are read from cache at approximately 0.1× the normal input token cost. This is particularly valuable for the ICP JSON, which can be 1,000–2,000 tokens.

---

## 5. What was deliberately not built

The following are out of scope, and their absence is a deliberate signal:

- **Authentication or multi-user support.** Adding a login screen does not improve the quality of the output.
- **Database persistence.** Each run is stateless. The sender profile lives in browser memory for the session.
- **CRM integration or email sending.** The pipeline ends at a human approval gate. This is a deliberate counter-positioning to Artisan's "replace humans" narrative — augmentation, not replacement.
- **Contact discovery or email-finding.** Out of scope per the original brief.
- **Account suggestion list.** Removed because it emitted company names from training memory without citations — the exact failure mode the anti-fabrication thesis is built against. If lookalike finding is added later, it must be search-grounded with source URLs.

---

## 6. Grounding audit

**Per-dimension citation enforcement.** After the `target_signals_and_fit` call, the agent verifies every cited evidence id against the retrieved chunk map. A dimension judgment of `present` or `partial` with no valid citation ids is downgraded to `absent` in code. This rule cannot be bypassed by the model's output.

**Claim map.** Every factual claim in the drafted emails must cite at least one evidence id. Claims where the cited id does not resolve to a retrieved chunk are marked `grounded=false` and rendered in red in the UI. This turns hallucinated personalisation from a silent failure into a visible one.

**Fit scoring — company-name independence.** `tests/test_scoring.py::test_same_dimensions_same_score_regardless_of_name` proves the deterministic formula is identical for two inputs with the same judgment pattern but different company names in the note field.

**Confidence degradation.** When the Mode 1 corpus is too thin (fewer than 2 retrieved chunks or under 400 total characters), the agent switches to a constrained prompt that instructs the model to leave any unsubstantiated dimension empty, marks `confidence='low'` on the returned profile, and keyword-scans the corpus to list specific data gaps. The user sees what was missing, not a confident but fabricated result.

---

## 7. Token economy

**Model routing.** The main model (sonnet) handles ICP synthesis, fit scoring, email drafting, and Critic review. The light model (haiku) handles customer enrichment, the adaptive planner, the Strategist, and all web research calls. Routing is explicit per call.

**Prompt caching.** System prompts are always marked cacheable. The ICP JSON and sender value proposition are passed as cached prefixes on Mode 2 calls.

**Evidence budget.** Each evidence snippet is truncated to `EVIDENCE_SNIPPET_CHARS` (default 400 chars) and the total evidence sent to any single call is capped at `EVIDENCE_BUDGET_CHARS` (default 6,000 chars). Highest-BM25-ranking chunks are prioritised when the budget is exceeded.

**Token meter.** `llm.py::TokenMeter` tracks input tokens, output tokens, cache write tokens, cache read tokens, elapsed time, and per-task breakdowns for every run. The snapshot is emitted with the final event of each mode so token usage is visible in the UI.

---

## 8. Known limitations

**Qualifying questions regenerate each session.** Mode 1 produces qualifying questions via an LLM call with no temperature control. They are stable within a session but vary across sessions. Profile persistence to disk (not yet built) would freeze the ICP including qualifying questions, making Mode 2 scores consistent across browser reloads.

**Scoring is non-deterministic.** The pain fit score is an LLM judgment and will vary across runs with the same input. The deterministic dimension score mitigates this for the auditable layer, but the headline verdict will shift. Reducing temperature on the fit call or running multiple samples and averaging would improve consistency.

**BM25 misses paraphrase.** If a target site uses vocabulary that does not match the query keywords, relevant chunks will not be retrieved. Hybrid retrieval (BM25 + embedding reranker) would improve recall without changing the architecture.

**Web search results vary by session.** Tavily returns slightly different results across runs for the same query (ranking shifts, new articles). This is expected — the pipeline is designed to work with any credible evidence set, not a fixed one. If exact reproducibility matters, persist the raw snippets alongside the profile.

**No signal freshness ranking.** Detected buying signals are not dated relative to each other. A funding round from 2021 and one from 2025 are treated with equal weight. Date extraction and recency scoring would let signals be ranked by timeliness.

**No profile persistence.** The sender profile lives in browser memory and is lost on page reload. Persisting profiles to disk would eliminate the need to re-run Mode 1 for each session, fix qualifying question instability, and reduce API cost.

**Critic rewrites are not verified.** If the Critic rewrites an email, the rewritten version is used without a second grounding check. A rewrite could introduce new claims that cite non-existent evidence ids. A second citation verification pass after Critic output would close this gap.
