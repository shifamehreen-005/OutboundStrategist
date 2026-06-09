"""Runtime configuration — reads environment variables once at import time.

All tuneable knobs live here so they can be changed without touching pipeline
logic. Keys are read from the process environment; a local .env file is loaded
automatically (via python-dotenv) so ANTHROPIC_API_KEY and TAVILY_API_KEY
work without shell exports.

MOCK_MODE is enabled automatically when ANTHROPIC_API_KEY is absent, so the
app always runs end-to-end with no credentials required.
"""
from __future__ import annotations

import os

# Load a local .env (if present) so keys like ANTHROPIC_API_KEY / TAVILY_API_KEY
# work without exporting them in the shell. Existing shell vars take precedence
# (load_dotenv does not override). No-op if python-dotenv isn't installed.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # noqa: BLE001 — dotenv optional
    pass


class Config:
    # If no API key is present we fall back to MOCK_MODE automatically so the
    # app always runs (great for a demo / reviewer with no key).
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    MOCK_MODE: bool = (
        os.getenv("MOCK_MODE", "").lower() in {"1", "true", "yes"}
        or not os.getenv("ANTHROPIC_API_KEY", "")
    )

    # Tavily web search — deterministic, stable results. When this key is set,
    # all web-search calls go through Tavily (search + LLM extraction) instead
    # of Claude's native web_search tool. Falls back to native if unset.
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # Main model — synthesis, scoring, email drafting.
    MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
    # Light model — signal extraction, suggestions, verification-style calls.
    MODEL_LIGHT: str = os.getenv("ANTHROPIC_MODEL_LIGHT", "claude-haiku-4-5-20251001")

    # Retrieval knobs ------------------------------------------------------
    CHUNK_CHARS: int = int(os.getenv("CHUNK_CHARS", "700"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    TOP_K: int = int(os.getenv("TOP_K", "6"))           # snippets per sub-query
    MAX_SNIPPET_CHARS: int = int(os.getenv("MAX_SNIPPET_CHARS", "240"))
    # Evidence sent to Claude: each chunk is truncated to this many chars
    # (0 = no cap / full chunk). Lower = fewer input tokens; raise if quality drops.
    EVIDENCE_SNIPPET_CHARS: int = int(os.getenv("EVIDENCE_SNIPPET_CHARS", "400"))
    # Hard cap on total evidence chars per call; excess chunks are dropped,
    # keeping the highest-BM25-priority ones first.
    EVIDENCE_BUDGET_CHARS: int = int(os.getenv("EVIDENCE_BUDGET_CHARS", "6000"))

    # Fetcher --------------------------------------------------------------
    FETCH_TIMEOUT: int = int(os.getenv("FETCH_TIMEOUT", "15"))
    # If a statically-fetched page yields fewer than this many chars of text,
    # retry once with the Playwright JS-render fallback (if installed).
    JS_RENDER_THRESHOLD: int = int(os.getenv("JS_RENDER_THRESHOLD", "600"))
    # Default crawl budgets per mode.  Lower = fewer noise pages + faster runs.
    # The /customers page is fetched separately by find_customer_links so it
    # doesn't consume from the Mode 1 budget.
    MAX_PAGES_SENDER: int = int(os.getenv("MAX_PAGES_SENDER", "4"))
    MAX_PAGES_TARGET: int = int(os.getenv("MAX_PAGES_TARGET", "5"))
    USER_AGENT: str = os.getenv(
        "USER_AGENT",
        "OutboundStrategistBot/1.0 (+research; respects robots)",
    )
    # If total text from a target's own site is below this threshold (chars),
    # the agent also fetches YC and Product Hunt for richer signal/fit evidence.
    ENRICH_THRESHOLD: int = int(os.getenv("ENRICH_THRESHOLD", "2000"))
    # Customer discovery: use Claude's native web search (+ DDG fallback) to
    # find a sender's customers across the open web, not just their own site.
    # Default ON in live mode (auto no-op in MOCK_MODE). Set to "0" to disable
    # (site-only) if you want to avoid web-search cost.
    CUSTOMER_WEB_SEARCH: bool = (
        os.getenv("CUSTOMER_WEB_SEARCH", "1").lower() in {"1", "true", "yes"}
    )
    CUSTOMER_WEB_MAX_SEARCHES: int = int(os.getenv("CUSTOMER_WEB_MAX_SEARCHES", "5"))


config = Config()
