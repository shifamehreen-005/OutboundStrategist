# Outbound Strategist

An evidence-grounded agentic outbound tool. Provide a sender's website and it builds a structured Ideal Customer Profile derived from real named customers. Provide a target account and a buyer persona and it scores fit, extracts buying signals, and produces two cited outbound emails — every factual claim linked to the retrieved snippet that supports it.

The pipeline is multi-agent by design: each agent has a single responsibility, its own system prompt, a typed output contract, and the cheapest model that can do the job. No monolithic prompt; no framework dependency; each stage is one file.

> Architecture decisions, grounding rationale, and known limitations live in [`DECISIONS.md`](DECISIONS.md).

---

## Thesis

Artisan's AI BDR (Ava) has two widely documented weaknesses: templated output despite personalisation claims, and single-agent shallowness. This tool is built directly against both:

- **Grounded personalisation** — every email claim must cite a retrieved evidence snippet or it is flagged unverified. Fabricated personalisation is a visible failure, not a silent one.
- **Specialised multi-agent pipeline** — Researcher, Signal Hunter, Fit Scorer, Strategist, Drafter, and Critic each do one job with dedicated prompts and model tiers, producing structurally different emails rather than reworded templates.

---

## Run

### 1 — Create and activate the virtual environment

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Install Playwright (JS-render fallback)

```bash
playwright install chromium
```

Required for JS-heavy sites that return near-empty static HTML. Degrades gracefully if absent.

### 4 — Start the server

```bash
uvicorn backend.app:app --port 8000
```

> **Tip for development:** add `--reload` so Python file changes take effect without restarting:
> ```bash
> uvicorn backend.app:app --port 8000 --reload
> ```
> CSS/JS changes are always live (just hard-refresh `Ctrl+Shift+R`). Python changes require a server restart or `--reload`.

Confirm it is running:

```bash
curl http://localhost:8000/api/health
# {"ok": true, "mock_mode": true,  "model": "claude-sonnet-4-5"}   ← mock mode - No API Key
# {"ok": true, "mock_mode": false, "model": "claude-sonnet-4-5"}   ← live mode - API Key
```

Open <http://localhost:8000>.

### 5 — Use it

**Mode 01 — Sender → ICP**

Enter a sender domain (e.g. `artisan.co`) and click *Run agent*. The pipeline streams step-by-step: fetch → enrich → customers → synthesise. The output shows:

- Value proposition and one-liner
- Named real customers extracted from the site and web, each with an industry tag, size badge, and provenance label (site / case study / web)
- The customer pattern — the common profile across actual buyers, which the ICP is derived from
- Full ICP: industries, size bands, buying triggers (each cited), buyer personas (each cited), anti-patterns, and qualifying questions Mode 2 will answer

**Mode 02 — Target → fit + outbound**

The sender profile from Mode 01 is passed automatically. Enter a target domain (e.g. `revolut.com`) and a buyer persona (e.g. `VP of Sales`). The pipeline runs:

- Web search for recent signals (funding, hiring, launches, leadership changes)
- Fit scoring: LLM-judged pain fit (0–100), worth reaching out (yes/maybe/no), best outreach angle, and answers to each qualifying question from the ICP
- Two structurally different emails produced by Strategist → Drafter → Critic
- Evidence panel: every cited E-id links to the source URL and snippet

Hover any `E#` chip to highlight its source snippet in the evidence panel.

### 6 — Run tests

```bash
pytest -q
# 23 passed — no API key needed; all tests run in mock mode
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(unset)* | Required for live mode; absent triggers mock mode automatically |
| `TAVILY_API_KEY` | *(unset)* | When set, all web search runs through Tavily (stable, deterministic) instead of Claude's native web search |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-5` | Main model — synthesis, fit scoring, email drafting, Critic |
| `ANTHROPIC_MODEL_LIGHT` | `claude-haiku-4-5-20251001` | Light model — customer enrichment, Strategist, planner, web research |
| `MOCK_MODE` | `0` | Force mock mode even when an API key is present |
| `CUSTOMER_WEB_SEARCH` | `1` | Enable web search for customer discovery in Mode 1 (live mode only) |
| `CUSTOMER_WEB_MAX_SEARCHES` | `5` | Maximum web searches per Mode 1 customer discovery run |
| `JS_RENDER_THRESHOLD` | `600` | Character count below which Playwright JS-render fallback fires |
| `ENRICH_THRESHOLD` | `2000` | Character count below which YC/Product Hunt enrichment fires |

**Troubleshooting:** if the server reports a model-not-found error on startup, set `ANTHROPIC_MODEL` to a model name your account supports.

---

## Architecture

### Mode 01 — Sender → ICP

```
 URL (e.g. artisan.co)
        │
        ▼
  ┌─────────────┐
  │  fetcher.py │  Crawl root + up to 4 prioritised pages
  │             │  (product, pricing, customers, about)
  │             │  Playwright JS-render fallback if text < 600 chars
  └──────┬──────┘
         │
         │  [if total site text < 2,000 chars]
         │         ▼
         │   ┌──────────────────────────────────┐
         │   │  Thin-site enrichment            │
         │   │  1. YC + Product Hunt static pages│
         │   │  2. Web search: 3 targeted queries│
         │   │     positioning / funding / reviews│
         │   └──────────────────────────────────┘
         │
         ▼
  ┌─────────────┐
  │  chunker.py │  ~700-char overlapping passages, heading-aware
  └──────┬──────┘
         │
         ▼
  ┌──────────────┐
  │ retriever.py │  BM25 index → 5 ICP queries + 3 customer queries
  └──────┬───────┘
         │
         ▼
  ┌──────────────────────────────────────────────────────────┐
  │  Customer discovery (runs BEFORE ICP synthesis)         │
  │  Source 1: case-study link slugs (site scrape)          │
  │  Source 2: page text extraction (site scrape)           │
  │  Source 3: web search — open web, press, review sites   │
  │  → merge + dedup → enrich all with industry + size tags │
  │    (uses source URL as context for ambiguous names)     │
  └──────────────────────────────┬───────────────────────────┘
                                 │ named_customers[]
                                 ▼
  ┌───────────────────────────────────────────────────────────┐
  │  ICP Synthesis — SENDER_SYSTEM (sonnet)                  │
  │  Reads E1…En + complete customer list                    │
  │  Industries derived from ACTUAL buyers, not marketing    │
  │  copy. Every trigger and buyer must cite evidence_ids.   │
  │  → value_proposition, ICP, qualifying_questions (3–5)   │
  └───────────────────────────────────────────────────────────┘

  [if < 4 retrieved chunks or low-confidence first pass]
  → Adaptive Planner (haiku): generates tailored BM25 queries,
    re-retrieves, re-synthesises once. Zero cost on rich sites.
```

### Mode 02 — Target → fit + outbound

```
 URL (e.g. revolut.com)  +  SenderProfile from Mode 01
        │
        ▼
  ┌─────────────┐
  │  fetcher.py │  Crawl target site (403/429 → graceful continue)
  │             │  YC + Product Hunt enrichment if site is sparse
  └──────┬──────┘
         │
         ▼
  ┌──────────────────────────────────────────┐
  │  Signal Hunter — web search (haiku)     │
  │  6 queries: funding, hiring, news,      │
  │  leadership, overview + broad search.   │
  │  Always runs, not just on sparse sites. │
  │                                         │
  │  Backend routing (llm.web_search_json): │
  │  • Tavily API (if TAVILY_API_KEY set)   │
  │    — 6 searches run concurrently,       │
  │      model extracts JSON from results   │
  │  • Claude native web_search tool        │
  │    (fallback, with truncation salvage)  │
  └──────────────────┬───────────────────────┘
                     │ web snippets added to corpus
                     ▼
  ┌──────────────┐
  │ retriever.py │  BM25 over merged corpus (site + web snippets)
  │              │  signal queries ∪ fit queries → ~19 snippets
  └──────┬───────┘
         │
         ▼
  ┌────────────────────────────────────────────────────────────┐
  │  Signals + Fit Scorer — SIGNALS_AND_FIT_SYSTEM (sonnet)   │
  │  One call. Evidence sent once. ICP + known customers       │
  │  in cache_prefix (prompt-cached; 0.1× cost on 2nd+ run). │
  │                                                            │
  │  Returns:                                                  │
  │  • signals[] — each citing ≥1 E-id                        │
  │  • dimensions — industry/trigger/buyer/size/geography      │
  │    strict: present/partial must cite ≥1 id; absent → []   │
  │  • qualification — ICP qualifying questions answered       │
  │    yes/partial/no/unknown, each with rationale + citation  │
  │  • pain_fit 0–100 — holistic LLM judgment                 │
  │  • worth_reaching_out — yes/maybe/no                      │
  │  • outreach_angle — best realistic angle for a rep        │
  └───────────────────────────────────┬────────────────────────┘
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
              dimensions dict              pain_fit (headline)
                         │
              ┌──────────▼──────────┐
              │   scoring.py        │
              │   deterministic     │  ← transparency check only;
              │   score 0–100       │    not the primary verdict
              └─────────────────────┘
                                      │
                                      ▼
  ┌──────────────────────────────────────────┐
  │  Strategist (haiku)                     │
  │  Picks #1 pain + #1 trigger.            │
  │  Writes a 2-sentence brief for each     │
  │  angle so the Drafter has a clear       │
  │  starting point — not a blank canvas.   │
  └──────────────────┬───────────────────────┘
                     │ AnglePlan
                     ▼
  ┌──────────────────────────────────────────┐
  │  Drafter — EMAIL_SYSTEM (sonnet)        │
  │  Pain-led email (PAS structure)         │
  │  Trigger-led email (event-first)        │
  │  House rules: ≤125 words, ≤4-word       │
  │  subject, no em dashes, no banned       │
  │  openers, outcomes not features,        │
  │  social proof from known customers.     │
  │  Every claim must cite ≥1 E-id.         │
  └──────────────────┬───────────────────────┘
                     │ draft emails
                     ▼
  ┌──────────────────────────────────────────┐
  │  Critic — CRITIC_SYSTEM (sonnet)        │
  │  16-point checklist. Rewrites any       │
  │  email that fails. If all pass,         │
  │  returns originals untouched.           │
  └──────────────────┬───────────────────────┘
                     │ final emails
                     ▼
          ┌───────────────────────┐
          │  TargetReport         │
          │  fit (pain_fit,       │
          │    qualification,     │
          │    dimension grid)    │
          │  signals[]            │
          │  emails[] + claims[]  │
          │  evidence[] — union   │
          └───────────────────────┘  of all cited E-ids
                                     with URL + snippet
```

---

## File map

```
outbound-strategist/
│
├── backend/
│   ├── app.py          FastAPI server — two NDJSON streaming endpoints:
│   │                   POST /api/sender  →  run_sender()
│   │                   POST /api/target  →  run_target()
│   │
│   ├── agent.py        Pipeline orchestration for both modes.
│   │                   Mode 1: fetch → enrich → customers → synthesise → replan?
│   │                   Mode 2: fetch → web search → BM25 → fit → strategist
│   │                           → drafter → critic → report
│   │
│   ├── fetcher.py      Polite HTTP crawler. Playwright JS-render fallback.
│   │                   Graceful 403/429 handling (continues with web search).
│   │                   YC + Product Hunt enrichment for sparse sites.
│   │
│   ├── chunker.py      ~700-char overlapping passages, heading-aware.
│   │                   Each chunk carries source URL, page title, section.
│   │
│   ├── retriever.py    BM25Okapi index. multi_search() unions results,
│   │                   deduplicates by chunk id, caps 3 results per URL.
│   │
│   ├── scoring.py      Deterministic fit formula — no LLM call.
│   │                   Weights: industry 25, trigger 25, buyer 25,
│   │                   size 15, geography 10. AP penalty: 20 pts each, cap 40.
│   │                   Used as a transparency check alongside the LLM verdict.
│   │
│   ├── prompts.py      All system + user prompt templates.
│   │                   Strategist, Drafter, Critic, Fit Scorer, ICP Synthesis,
│   │                   Customer Extraction, Customer Enrichment, Web Research.
│   │
│   ├── llm.py          Thin Claude wrapper. Prompt caching on system prompts
│   │                   and static prefixes (ICP JSON, value prop). TokenMeter
│   │                   tracks tokens per task and per model. MOCK_MODE bypasses
│   │                   the API entirely.
│   │
│   ├── schemas.py      Pydantic models for all inputs and outputs.
│   │                   Key types: SenderProfile, ICP, NamedCustomer,
│   │                   FitScore (pain_fit + qualification + dimensions),
│   │                   AnglePlan, CriticVerdict, Email, Claim, Evidence.
│   │
│   ├── mocks.py        Content-derived mock responses for MOCK_MODE.
│   │                   target_signals_and_fit, target_strategist, target_critic,
│   │                   sender_customers, sender_web_research all have branches.
│   │                   Mocks react to evidence content, not company names.
│   │
│   ├── config.py       Reads env vars: API key, model names, fetch timeout,
│   │                   chunk size, JS-render threshold, enrichment threshold,
│   │                   customer web-search toggle.
│   │
│   ├── browser.py      Playwright JS-render fallback. Optional dependency —
│   │                   degrades gracefully if Playwright is not installed.
│   │
│   └── search.py       Web search backends. Tavily is primary (concurrent
│                       queries via POST API). DuckDuckGo is the fallback
│                       for customer discovery when Tavily is not configured.
│                       Both return [] on any failure — callers degrade
│                       gracefully. Routing lives in llm.web_search_json().
│
├── frontend/
│   ├── index.html      Single-page UI — no build step, no bundler.
│   ├── app.js          Streams NDJSON; renders ICP, fit card (pain_fit gauge,
│   │                   worth badge, outreach angle, qualifying questions,
│   │                   collapsible dimension grid), emails, evidence panel.
│   └── styles.css      Dark theme, IBM Plex, amber accents.
│
├── tests/
│   ├── conftest.py         MOCK_MODE=1, project root on sys.path
│   ├── test_mode1.py       Mode 1: happy path, thin corpus, confidence=low,
│   │                       grounding coverage, named customers, qualifying Qs
│   ├── test_mode2.py       Mode 2: dimension grounding rules, score == formula
│   └── test_scoring.py     Pure unit tests for scoring.py — no LLM, no network.
│                           Proves score is deterministic and company-name-blind.
│
├── DECISIONS.md        Architecture and design decisions with rationale.
├── requirements.txt
└── README.md           This file.
```
