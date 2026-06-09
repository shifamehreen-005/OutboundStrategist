"""Anthropic API wrapper with prompt caching, token metering, and web search.

Two public methods:
  json()             — single-turn JSON extraction call with prompt caching.
  web_search_json()  — web search + JSON extraction, with three backends:
                         1. MOCK_MODE: mock fixtures from mocks.py
                         2. Tavily (when TAVILY_API_KEY is set and queries are
                            provided): concurrent searches via search.tavily_search(),
                            then LLM extracts JSON from results. Deterministic and
                            immune to max_tokens truncation issues.
                         3. Claude native web_search tool: the model runs its own
                            searches. A salvage helper recovers complete array items
                            even if the output is truncated at max_tokens.

Prompt caching: system prompts are always wrapped in an ephemeral cached block.
Callers may also pass a cache_prefix (e.g. the ICP JSON) as a second cached
block that sits before the variable user message — billed at 0.1× input cost
on repeated calls within a session's 5-minute cache TTL.

TokenMeter tracks input/output/cache tokens per task and per model for every
call in a run. The snapshot is emitted with the final agent event so token
usage and cost are visible in the UI footer.

MOCK_MODE: bypasses the API entirely; caching is a no-op.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from .config import config
from . import mocks


class TokenMeter:
    def __init__(self) -> None:
        self.input = 0
        self.output = 0
        self.calls = 0
        self.cache_write = 0        # tokens written to cache (billed 1.25× input)
        self.cache_read  = 0        # tokens read from cache (billed 0.1× input)
        self.by_model: Dict[str, int] = {}
        self.by_task: Dict[str, Dict[str, Any]] = {}
        self._t0: float = time.monotonic()

    def add(self, i: int, o: int, model: str = "", task: str = "",
            cache_write: int = 0, cache_read: int = 0) -> None:
        self.input       += i
        self.output      += o
        self.calls       += 1
        self.cache_write += cache_write
        self.cache_read  += cache_read
        if model:
            self.by_model[model] = self.by_model.get(model, 0) + 1
        if task:
            entry = self.by_task.setdefault(task, {
                "input": 0, "output": 0, "model": model,
                "cache_write": 0, "cache_read": 0,
            })
            entry["input"]       += i
            entry["output"]      += o
            entry["cache_write"] += cache_write
            entry["cache_read"]  += cache_read
            if model:
                entry["model"] = model

    def snapshot(self) -> Dict[str, Any]:
        return {
            "input_tokens":       self.input,
            "output_tokens":      self.output,
            "calls":              self.calls,
            "elapsed_ms":         int((time.monotonic() - self._t0) * 1000),
            "cache_write_tokens": self.cache_write,
            "cache_read_tokens":  self.cache_read,
            "by_model":           dict(self.by_model),
            "by_task":            {k: dict(v) for k, v in self.by_task.items()},
        }


class LLM:
    def __init__(self) -> None:
        self.meter = TokenMeter()
        self._client = None
        if not config.MOCK_MODE:
            import anthropic
            self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    def json(self, system: str, user: str, task: str,
             max_tokens: int = 1500,
             model: Optional[str] = None,
             cache_prefix: str = "") -> Dict[str, Any]:
        """Return a parsed JSON object from Claude (or a mock fixture).

        system      — system prompt text; always wrapped in a cached content block.
        user        — variable part of the user message; never cached.
        cache_prefix— optional static prefix placed before `user` in a separate
                      cached content block. Use for content that is identical across
                      multiple calls within a run (e.g. ICP JSON, sender value prop).
        """
        resolved = model or config.MODEL

        if config.MOCK_MODE:
            self.meter.add(len(user) // 4, 300, model="mock", task=task)
            return mocks.respond(task, user)

        json_instruction = "\n\nCompact JSON only — no prose, no markdown, no code fences."

        # System prompt as a single cached block.
        system_blocks: List[Dict[str, Any]] = [{
            "type": "text",
            "text": system + json_instruction,
            "cache_control": {"type": "ephemeral"},
        }]

        # User content: optional cached prefix + variable user message.
        if cache_prefix:
            user_content: List[Dict[str, Any]] = [
                {"type": "text", "text": cache_prefix,
                 "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": user},
            ]
        else:
            user_content = [{"type": "text", "text": user}]

        msg = self._client.messages.create(
            model=resolved,
            max_tokens=max_tokens,
            system=system_blocks,
            messages=[{"role": "user", "content": user_content}],
        )

        cw = getattr(msg.usage, "cache_creation_input_tokens", 0) or 0
        cr = getattr(msg.usage, "cache_read_input_tokens",       0) or 0
        self.meter.add(msg.usage.input_tokens, msg.usage.output_tokens,
                       model=resolved, task=task, cache_write=cw, cache_read=cr)

        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        try:
            return _parse_json(text)
        except (json.JSONDecodeError, ValueError):
            # One retry with a higher token ceiling — the usual cause is the
            # response being truncated mid-JSON at max_tokens.
            retry = self._client.messages.create(
                model=resolved,
                max_tokens=min(int(max_tokens * 1.6) + 400, 4096),
                system=system_blocks,
                messages=[{"role": "user", "content": user_content}],
            )
            self.meter.add(retry.usage.input_tokens, retry.usage.output_tokens,
                           model=resolved, task=task)
            rtext = "".join(b.text for b in retry.content
                            if getattr(b, "type", "") == "text")
            return _parse_json(rtext)

    def web_search_json(self, system: str, user: str, task: str,
                        max_tokens: int = 900, max_searches: int = 3,
                        model: Optional[str] = None,
                        queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run a web search and return parsed JSON.

        Two backends:
        - Tavily (when TAVILY_API_KEY is set AND `queries` provided): we run the
          searches ourselves and the model only EXTRACTS the JSON from results.
          Deterministic and stable — no native-tool truncation issues.
        - Claude native web_search tool (fallback): the model runs its own
          searches. Robust to max_tokens truncation via array salvage.

        Use for facts that live on the open web. Every item carries a source_url.
        Raises on API error so the caller can fall back to another source.
        """
        resolved = model or config.MODEL_LIGHT
        if config.MOCK_MODE:
            self.meter.add(len(user) // 4, 300, model="mock", task=task)
            return mocks.respond(task, user)

        if config.TAVILY_API_KEY and queries:
            return self._tavily_search_json(system, user, task, queries,
                                            max_tokens, resolved)

        msg = self._client.messages.create(
            model=resolved,
            max_tokens=max_tokens,
            system=system + "\n\nEnd your reply with ONLY the JSON object requested.",
            messages=[{"role": "user", "content": user}],
            tools=[{"type": "web_search_20250305", "name": "web_search",
                    "max_uses": max_searches}],
        )
        cw = getattr(msg.usage, "cache_creation_input_tokens", 0) or 0
        cr = getattr(msg.usage, "cache_read_input_tokens",       0) or 0
        self.meter.add(msg.usage.input_tokens, msg.usage.output_tokens,
                       model=resolved, task=task, cache_write=cw, cache_read=cr)
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        if not text.strip():
            return {}
        try:
            return _parse_json(text)
        except json.JSONDecodeError:
            # Web search + JSON can hit max_tokens and truncate mid-object,
            # leaving the closing braces off. Salvage the complete array items
            # that DID finish rather than dropping all evidence.
            for key in ("snippets", "customers"):
                items = _salvage_array_objects(text, key)
                if items:
                    return {key: items}
            return {}

    def _tavily_search_json(self, system: str, user: str, task: str,
                            queries: List[str], max_tokens: int,
                            model: str) -> Dict[str, Any]:
        """Run `queries` through Tavily, then have the model extract the
        requested JSON from the collected results (no native web tool)."""
        from concurrent.futures import ThreadPoolExecutor
        from . import search as web

        # Run all Tavily searches concurrently (each is a blocking HTTP call),
        # then merge in query order so the result set stays deterministic.
        with ThreadPoolExecutor(max_workers=len(queries) or 1) as pool:
            per_query = list(pool.map(lambda q: web.tavily_search(q, limit=4),
                                      queries))

        results = []
        seen = set()
        for group in per_query:
            for r in group:
                if r.url in seen:
                    continue
                seen.add(r.url)
                results.append(r)

        if not results:
            return {}

        block = "\n\n".join(
            f"[{i}] {r.title}\nURL: {r.url}\n{r.snippet[:500]}"
            for i, r in enumerate(results, 1)
        )
        # Reuse the caller's user prompt (which carries the JSON schema) but feed
        # the Tavily results in place of the model running its own searches.
        extract_user = (
            f"{user}\n\n"
            "Do NOT run your own searches. Use ONLY the SEARCH RESULTS below. "
            "For every fact, set source_url to the result's URL it came from. "
            "Omit anything the results don't support.\n\n"
            f"SEARCH RESULTS:\n{block}"
        )
        return self.json(system, extract_user, task=task,
                         max_tokens=max_tokens, model=model)


def _balanced_object(text: str) -> Optional[str]:
    """Return the first top-level {...} object via brace matching (handles
    strings/escapes), so a trailing stray token doesn't break extraction."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return None  # unbalanced (e.g. truncated) — caller will retry


def _salvage_array_objects(text: str, key: str) -> list:
    """Recover the complete `{...}` items inside `"key": [ ... ]` from text that
    was truncated before the array/object could close (max_tokens cutoff).
    Returns the items that fully parsed; ignores the trailing partial one."""
    m = re.search(rf'"{key}"\s*:\s*\[', text)
    if not m:
        return []
    i = m.end()
    items: list = []
    while i < len(text):
        # advance to the next object start (or bail at the array close)
        while i < len(text) and text[i] not in "{]":
            i += 1
        if i >= len(text) or text[i] == "]":
            break
        obj = _balanced_object(text[i:])
        if not obj:
            break  # truncated partial object — stop here
        try:
            items.append(json.loads(re.sub(r",(\s*[}\]])", r"\1", obj)))
        except json.JSONDecodeError:
            break
        i += len(obj)
    return items


def _parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    candidate = _balanced_object(text) or text
    # Try as-is, then with trailing commas stripped (a common LLM slip).
    for attempt in (candidate, re.sub(r",(\s*[}\]])", r"\1", candidate)):
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            continue
    # Re-raise the original error for the caller's retry path.
    return json.loads(candidate)
