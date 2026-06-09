"""Lightweight, polite crawler.

Strategy: fetch the root page, then follow a *small, prioritised* set of
same-domain links that tend to carry ICP / buying-signal information
(product, pricing, customers, about, news, blog). We never crawl the
whole site -- recall here is cheap to get wrong and expensive in tokens,
so we bias hard toward a handful of high-value pages.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set
from urllib.parse import urljoin, urlparse

import httpx
from selectolax.parser import HTMLParser

from .config import config
from . import browser

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@dataclass
class Page:
    url: str
    title: str
    text: str


def _domain(url: str) -> str:
    netloc = urlparse(url if url.startswith("http") else "https://" + url).netloc
    return netloc.split(":")[0].replace("www.", "").lower()


def _load_fixture(url: str) -> Optional[List[Page]]:
    """If a bundled fixture exists for this domain, use it (offline demo)."""
    fp = FIXTURE_DIR / f"{_domain(url)}.json"
    if fp.exists():
        data = json.loads(fp.read_text())
        return [Page(**p) for p in data]
    return None


# Link path keywords ranked by how useful they usually are for ICP + signals.
# Common paths probed when link discovery is sparse (JS-rendered / SPA sites).
# Tried in order; stops once max_pages is reached.
_PROBE_PATHS = [
    "/about", "/about-us", "/product", "/platform",
    "/pricing", "/customers", "/blog", "/company",
]

PRIORITY_PATHS = [
    "product", "platform", "solution", "feature",
    "pricing", "plans",
    "customer", "case-stud", "stories", "testimonial",
    "about", "company", "team",
    "news", "press", "newsroom", "blog", "announc",
    "industr", "use-case",
]

SKIP_PATTERNS = re.compile(
    r"(login|signin|sign-in|signup|sign-up|privacy|terms|cookie|dpa|legal|"
    r"\.pdf$|\.png$|\.jpg$|\.svg$|\.webp$|mailto:|tel:|#)",
    re.I,
)


def _clean_text(html: str) -> tuple[str, str]:
    """Extract title + readable body text, preserving section headings.

    Headings (h1–h4) are marked with a leading "## " so the chunker can carry
    section context into each chunk — improves BM25 recall and citation readability.
    """
    tree = HTMLParser(html)
    title = ""
    if tree.css_first("title"):
        title = tree.css_first("title").text(strip=True)
    for sel in ["script", "style", "noscript", "svg", "nav", "footer", "header"]:
        for node in tree.css(sel):
            node.decompose()
    headings = set()
    for h in tree.css("h1, h2, h3, h4"):
        t = h.text(strip=True)
        if t:
            headings.add(t)
    body = tree.body or tree
    text = body.text(separator="\n", strip=True) if body else ""
    if headings:
        marked = []
        for line in text.split("\n"):
            s = line.strip()
            marked.append(f"## {s}" if s in headings else line)
        text = "\n".join(marked)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return title, text


def _same_domain(root: str, link: str) -> bool:
    return urlparse(root).netloc.split(":")[0].replace("www.", "") == \
        urlparse(link).netloc.split(":")[0].replace("www.", "")


def _rank_link(path: str) -> int:
    p = path.lower()
    for i, kw in enumerate(PRIORITY_PATHS):
        if kw in p:
            return i
    return len(PRIORITY_PATHS) + 1


def discover_links(root_url: str, html: str, limit: int) -> List[str]:
    tree = HTMLParser(html)
    found: List[str] = []
    seen: Set[str] = set()
    for a in tree.css("a[href]"):
        href = a.attributes.get("href") or ""
        if not href or SKIP_PATTERNS.search(href):
            continue
        full = urljoin(root_url, href).split("#")[0].rstrip("/")
        if full in seen or not _same_domain(root_url, full):
            continue
        if full.rstrip("/") == root_url.rstrip("/"):
            continue
        seen.add(full)
        found.append(full)
    found.sort(key=lambda u: _rank_link(urlparse(u).path))
    return found[:limit]


def _company_slug(domain: str) -> str:
    """autosana.ai → autosana, my-co.com → my-co."""
    return _domain(domain).split(".")[0]


# Named customers usually live in case-study URLs (e.g. /customers/sumup-case-study),
# not in body text or logo-wall images. This is the most reliable customer source.
_CUSTOMER_LINK_RE = re.compile(
    r"/(?:customers?|case-stud(?:y|ies)|client-stories|stories|clients)/"
    r"([a-z0-9][a-z0-9\-]+)", re.I)
_SLUG_SUFFIXES = ("-case-study", "-case-studies", "-casestudy",
                  "-customer-story", "-story", "-stories", "-case")
_SLUG_STOP = {"", "all", "index", "case-study", "case-studies", "customer",
              "customers", "stories", "story", "client", "clients", "page"}


def _slug_hint(slug: str) -> str:
    s = slug.lower()
    for suf in _SLUG_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s.strip("-")


def discover_customer_links(root_url: str, html: str) -> List[dict]:
    """Parse <a href> for case-study / customer links. Returns [{hint, url}]."""
    tree = HTMLParser(html)
    out: List[dict] = []
    seen: Set[str] = set()
    for a in tree.css("a[href]"):
        href = a.attributes.get("href") or ""
        m = _CUSTOMER_LINK_RE.search(href)
        if not m:
            continue
        hint = _slug_hint(m.group(1))
        if hint in _SLUG_STOP or hint in seen:
            continue
        seen.add(hint)
        full = urljoin(root_url, href).split("?")[0].split("#")[0]
        out.append({"hint": hint, "url": full})
    return out


_CTA_RE = re.compile(
    r"\b(credit card|no credit|free trial|sign up|get started|book a demo"
    r"|schedule a call|start for free|try for free|get \$|credits?\.?\s*$"
    r"|hire ava|request a demo|watch a demo|talk to us)\b",
    re.I,
)


def _is_cta_line(line: str) -> bool:
    return bool(_CTA_RE.search(line))


def enrich_customer_snippets(links: List[dict],
                             max_each: int = 300) -> List[dict]:
    """Fetch each case-study URL and extract a short description snippet.

    Returns the same list with a 'snippet' key added to each dict where
    the fetch succeeded. Bounded: fetches at most 8 links, short timeout,
    never raises — a failed fetch just leaves snippet=''.
    """
    to_enrich = [l for l in links if l.get("url")][:8]
    headers = {"User-Agent": config.USER_AGENT}
    try:
        with httpx.Client(timeout=8.0, follow_redirects=True,
                          headers=headers) as client:
            for l in to_enrich:
                try:
                    r = client.get(l["url"])
                    if r.status_code != 200:
                        l["snippet"] = ""
                        continue
                    _, text = _clean_text(r.text)
                    # Skip CTA lines; strip ## heading markers but keep the
                    # content (case-study titles still describe the customer).
                    lines = [ln.strip().lstrip("#").strip()
                             for ln in text.split("\n")
                             if len(ln.strip()) > 40
                             and not _is_cta_line(ln)]
                    l["snippet"] = lines[0][:max_each] if lines else ""
                except Exception:  # noqa: BLE001
                    l["snippet"] = ""
    except Exception:  # noqa: BLE001
        pass
    # Ensure every link has the key
    for l in links:
        l.setdefault("snippet", "")
    return links


def find_customer_links(url: str, max_links: int = 12) -> List[dict]:
    """Live-fetch the homepage (+ /customers, /case-studies index) and extract
    named-customer links from case-study URLs. Returns [{hint, url}].

    This is the reliable customer source: a logo wall is images, but case-study
    pages live at stable /customers/<name> URLs. Returns [] on any failure."""
    if not url.startswith("http"):
        url = "https://" + url
    url = url.rstrip("/")
    headers = {"User-Agent": config.USER_AGENT}
    links: List[dict] = []
    try:
        with httpx.Client(timeout=config.FETCH_TIMEOUT, follow_redirects=True,
                          headers=headers) as client:
            for page in (url, url + "/customers", url + "/case-studies"):
                try:
                    r = client.get(page)
                    if r.status_code != 200:
                        continue
                    html = r.text
                    if len(html) < 2000 and browser.is_available():
                        rendered = browser.render_html(page)
                        if rendered:
                            html = rendered
                    links += discover_customer_links(url, html)
                except Exception:  # noqa: BLE001
                    continue
    except Exception:  # noqa: BLE001
        return []
    # Dedupe by hint, preserve order.
    seen: Set[str] = set()
    uniq: List[dict] = []
    for l in links:
        if l["hint"] in seen:
            continue
        seen.add(l["hint"])
        uniq.append(l)
    return uniq[:max_links]


def fetch_enrichment(domain: str) -> List[Page]:
    """Fetch YC and Product Hunt pages for companies whose own site is sparse.

    Returns whatever pages load successfully; silently skips 404s and timeouts.
    """
    slug = _company_slug(domain)
    sources = [
        f"https://www.ycombinator.com/companies/{slug}",
        f"https://www.producthunt.com/products/{slug}",
    ]
    pages: List[Page] = []
    headers = {"User-Agent": config.USER_AGENT}
    with httpx.Client(timeout=10.0, follow_redirects=True, headers=headers) as client:
        for src in sources:
            try:
                r = client.get(src)
                if r.status_code != 200:
                    continue
                title, text = _clean_text(r.text)
                if len(text) > 300:
                    pages.append(Page(url=src, title=title, text=text))
            except Exception:  # noqa: BLE001
                continue
    return pages


def _fetch_clean(client: httpx.Client, url: str) -> tuple[str, str, str]:
    """GET + clean a single URL. If the static text is thin and Playwright is
    available, re-render with JS and re-clean. Returns (title, text, raw_html).
    Raises on a hard HTTP failure (caller decides if that's fatal)."""
    r = client.get(url)
    r.raise_for_status()
    title, text = _clean_text(r.text)
    raw_html = r.text

    # JS-render fallback: the static fetch returned a near-empty shell.
    if len(text) < config.JS_RENDER_THRESHOLD and browser.is_available():
        rendered = browser.render_html(url)
        if rendered:
            r_title, r_text = _clean_text(rendered)
            if len(r_text) > len(text):
                title, text, raw_html = r_title, r_text, rendered
    return title, text, raw_html


def fetch_site(root_url: str, max_pages: int = 5,
               coverage_queries: List[str] = None) -> List[Page]:
    """Fetch root + up to (max_pages-1) prioritised internal pages.

    Stops early if BM25 coverage across `coverage_queries` is already full
    (all queries have at least one positive-scoring chunk), so we don't fetch
    noise pages when the first few pages already answer every question.

    Prefers a bundled offline fixture for the domain if one exists, so the
    demo runs with no network. Otherwise fetches live, with a Playwright
    JS-render fallback for pages that come back as empty JS shells.
    """
    fixture = _load_fixture(root_url)
    if fixture is not None:
        return fixture[:max_pages]

    if not root_url.startswith("http"):
        root_url = "https://" + root_url
    root_url = root_url.rstrip("/")

    # Lazy import to avoid circular dependency (chunker → fetcher → chunker).
    from .chunker import chunk_pages as _chunk_pages
    from .retriever import Retriever as _Retriever

    def _coverage(pages_so_far: List[Page]) -> float:
        """BM25 coverage of coverage_queries over current pages."""
        if not coverage_queries or not pages_so_far:
            return 0.0
        chunks = _chunk_pages(pages_so_far)
        if not chunks:
            return 0.0
        return _Retriever(chunks).coverage(coverage_queries)

    pages: List[Page] = []
    headers = {"User-Agent": config.USER_AGENT}
    with httpx.Client(
        timeout=config.FETCH_TIMEOUT, follow_redirects=True, headers=headers
    ) as client:
        try:
            title, text, root_html = _fetch_clean(client, root_url)
            pages.append(Page(url=root_url, title=title, text=text))
        except Exception as e:  # noqa: BLE001
            # 403/429/connection errors on the root page are non-fatal: web
            # search and YC/Product Hunt enrichment will cover the gap.
            # Re-raise only on truly unrecoverable errors (bad URL scheme etc).
            err = str(e).lower()
            if any(x in err for x in ("403", "404", "429", "connect", "timeout",
                                       "ssl", "forbidden", "too many")):
                root_html = ""   # no links to discover
            else:
                raise RuntimeError(f"Could not fetch {root_url}: {e}") from e

        for link in discover_links(root_url, root_html, limit=max_pages - 1):
            if len(pages) >= max_pages:
                break
            # Early-stop: all query dimensions already covered.
            if coverage_queries and _coverage(pages) >= 1.0:
                break
            try:
                lt, ltext, _ = _fetch_clean(client, link)
                if len(ltext) > 200:
                    pages.append(Page(url=link, title=lt, text=ltext))
            except Exception:  # noqa: BLE001
                continue

        # Probe common paths when link discovery was sparse (e.g. SPAs).
        if len(pages) < max_pages:
            if not (coverage_queries and _coverage(pages) >= 1.0):
                crawled = {p.url for p in pages}
                for path in _PROBE_PATHS:
                    if len(pages) >= max_pages:
                        break
                    if coverage_queries and _coverage(pages) >= 1.0:
                        break
                    probe = root_url + path
                    if probe in crawled:
                        continue
                    try:
                        pt, ptext, _ = _fetch_clean(client, probe)
                        if len(ptext) > 200:
                            pages.append(Page(url=probe, title=pt, text=ptext))
                            crawled.add(probe)
                    except Exception:  # noqa: BLE001
                        continue
    return pages
