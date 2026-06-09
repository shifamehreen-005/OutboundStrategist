"""Turn fetched pages into retrievable chunks.

Heading-aware: `_clean_text` (fetcher) marks section headings with a leading
"## ". The chunker tracks the current section, never lets a chunk span a
heading boundary, and carries the section into both the chunk title
("Page › Section") and the chunk body (prepended) so:
  - BM25 recall improves — heading words become matchable,
  - citations read cleanly — the claim map shows which section a fact came from,
  - chunks stay semantically coherent — one section per chunk.

Each chunk keeps its source URL so anything built from it (signals, claims,
fit rationale) can cite back to the exact page + section.
"""
from __future__ import annotations

import re
from typing import List

from .config import config
from .fetcher import Page
from .schemas import Chunk


def _split_sentences(text: str) -> List[str]:
    # Cheap sentence-ish split; good enough for chunk packing.
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _make_chunk(idx: int, page: Page, section: str, buf: str, pos: int) -> Chunk:
    title = f"{page.title} › {section}".strip(" ›") if section else page.title
    # Prepend the section so heading terms are retrievable from the body too.
    body = f"{section}. {buf}".strip() if section else buf
    return Chunk(id=f"c{idx}", url=page.url, title=title, text=body, position=pos)


def chunk_pages(pages: List[Page]) -> List[Chunk]:
    chunks: List[Chunk] = []
    size = config.CHUNK_CHARS
    overlap = config.CHUNK_OVERLAP
    idx = 0
    for page in pages:
        section = ""
        buf = ""
        pos = 0
        for line in page.text.split("\n"):
            s = line.strip()
            if not s:
                continue
            # Section boundary: flush the current buffer and switch section.
            if s.startswith("## "):
                if buf.strip():
                    chunks.append(_make_chunk(idx, page, section, buf, pos))
                    idx += 1
                    pos += 1
                    buf = ""
                section = s[3:].strip()
                continue
            for sent in _split_sentences(s):
                if len(buf) + len(sent) + 1 <= size:
                    buf = f"{buf} {sent}".strip()
                else:
                    if buf.strip():
                        chunks.append(_make_chunk(idx, page, section, buf, pos))
                        idx += 1
                        pos += 1
                    # carry a little overlap for cross-chunk context
                    tail = buf[-overlap:] if (overlap and buf) else ""
                    buf = f"{tail} {sent}".strip()
        if buf.strip():
            chunks.append(_make_chunk(idx, page, section, buf, pos))
            idx += 1
    return chunks
