"""Optional Playwright JS-render fallback.

Many modern marketing sites render their content in JavaScript, so a plain
HTTP GET returns a near-empty shell. When the static fetch yields too little
text, we re-fetch with a headless browser that executes JS.

Design rules:
  - Playwright is an OPTIONAL dependency. If it isn't installed (or the browser
    binary isn't present), `render_html` returns None and the caller falls back
    to the static text — the app never crashes for lack of a browser.
  - One browser launch per call; short timeout; networkidle wait so SPA content
    has a chance to paint.
"""
from __future__ import annotations

from typing import Optional

from .config import config

# Detect Playwright once at import. Absent → fallback is a silent no-op.
try:
    from playwright.sync_api import sync_playwright  # type: ignore
    _PLAYWRIGHT_AVAILABLE = True
except Exception:  # noqa: BLE001
    _PLAYWRIGHT_AVAILABLE = False


def is_available() -> bool:
    return _PLAYWRIGHT_AVAILABLE


def render_html(url: str, timeout_ms: int = 12000) -> Optional[str]:
    """Return fully-rendered HTML for `url`, or None if rendering is unavailable
    or fails. Never raises — a failed render is a non-fatal fallback miss."""
    if not _PLAYWRIGHT_AVAILABLE:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(user_agent=config.USER_AGENT)
                page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                html = page.content()
            finally:
                browser.close()
        return html
    except Exception:  # noqa: BLE001
        return None
