"""Web search backends: Tavily (primary) and DuckDuckGo (fallback).

Tavily is the primary backend when TAVILY_API_KEY is set. It provides clean,
structured results (title / url / content) via a simple POST API and is used
by all three web-search calls in the pipeline (sender customers, sender thin-
site enrichment, and Mode 2 target signal hunting).

DuckDuckGo is the secondary fallback used in one place only: customer
discovery in _web_customers() when Claude's native web_search tool is
unavailable and Tavily is not configured. Both backends return [] on any
failure so callers always degrade gracefully.

Routing in llm.web_search_json():
  MOCK_MODE            → mock fixtures (mocks.py)
  TAVILY_API_KEY + queries provided → tavily_search() + LLM extraction
  otherwise            → Claude native web_search tool
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import httpx
from selectolax.parser import HTMLParser

from .config import config


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""


def tavily_search(query: str, limit: int = 5,
                  search_depth: str = "basic") -> List[SearchResult]:
    """Search via Tavily's API. Returns up to `limit` results; [] on any
    failure (missing key, network error, bad response) so callers degrade.

    Tavily returns clean title/url/content for each hit, which we map straight
    onto SearchResult — no HTML scraping, so it's far more stable than DDG."""
    if not config.TAVILY_API_KEY:
        return []
    try:
        with httpx.Client(timeout=12.0) as client:
            r = client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": config.TAVILY_API_KEY,
                    "query": query,
                    "max_results": limit,
                    "search_depth": search_depth,
                },
            )
            r.raise_for_status()
            data = r.json()
    except Exception:
        return []

    results: List[SearchResult] = []
    for item in (data.get("results") or []):
        url = (item.get("url") or "").strip()
        title = (item.get("title") or "").strip()
        snippet = (item.get("content") or "").strip()
        if url.startswith("http") and (title or snippet):
            results.append(SearchResult(title=title, url=url, snippet=snippet))
    return results


def ddg_search(query: str, limit: int = 5) -> List[SearchResult]:
    """Return up to `limit` results for `query`; empty list on any failure."""
    try:
        with httpx.Client(
            timeout=8.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"},
        ) as client:
            r = client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query, "kl": "us-en"},
            )
            r.raise_for_status()
    except Exception:
        return []

    results: List[SearchResult] = []
    try:
        tree = HTMLParser(r.text)
        for div in tree.css("div.result"):
            a = div.css_first("a.result__a")
            snip_el = div.css_first("a.result__snippet")
            if not a:
                continue
            url = a.attributes.get("href", "")
            title = a.text(strip=True)
            snippet = snip_el.text(strip=True) if snip_el else ""
            if title and url and url.startswith("http"):
                results.append(SearchResult(title=title, url=url, snippet=snippet))
            if len(results) >= limit:
                break
    except Exception:
        return []

    return results


def icp_queries(icp_dict: dict) -> List[str]:
    """Build 2-4 search queries from an ICP dict without an extra LLM call."""
    industries = icp_dict.get("industries", [])[:2]
    triggers = [t.get("name", "") for t in icp_dict.get("triggers", [])[:2]]
    queries = []
    for ind in industries:
        queries.append(f"{ind} startup company")
    for trig in triggers:
        if trig:
            queries.append(f"B2B company {trig.lower()}")
    return queries or ["B2B software startup recently funded"]
