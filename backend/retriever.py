"""BM25 retrieval over the chunk corpus.

Why BM25 and not a vector DB? The corpus here is tiny -- a handful of
pages from one or two sites (tens of chunks). Lexical BM25 gives strong,
explainable recall on this scale with zero embedding cost and no second
API dependency. Claude is then used only to *read and ground* the small
set of candidates, which is where its judgement actually adds value.
This keeps token usage proportional to the answer, not the website.
"""
from __future__ import annotations

import re
from typing import List

from rank_bm25 import BM25Okapi

from .config import config
from .schemas import Chunk


_TOKEN = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> List[str]:
    return _TOKEN.findall(text.lower())


class Retriever:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self._corpus = [_tok(c.text) for c in chunks] or [[""]]
        self._bm25 = BM25Okapi(self._corpus)

    def search(self, query: str, top_k: int | None = None) -> List[Chunk]:
        top_k = top_k or config.TOP_K
        if not self.chunks:
            return []
        scores = self._bm25.get_scores(_tok(query))
        ranked = sorted(
            range(len(self.chunks)), key=lambda i: scores[i], reverse=True
        )
        out, seen_urls = [], {}
        for i in ranked:
            if scores[i] <= 0:
                continue
            c = self.chunks[i]
            # light per-URL cap so one page can't dominate the context
            seen_urls[c.url] = seen_urls.get(c.url, 0) + 1
            if seen_urls[c.url] > 3:
                continue
            out.append(c)
            if len(out) >= top_k:
                break
        return out

    def coverage(self, queries: List[str], per_query: int = 1) -> float:
        """Fraction of queries that return at least one positive-scoring chunk.

        Used for early-stop: if every dimension already has evidence, there is
        no point fetching another page.
        """
        if not self.chunks:
            return 0.0
        hits = sum(1 for q in queries if self.search(q, top_k=per_query))
        return hits / len(queries)

    def multi_search(self, queries: List[str], per_query: int = 4,
                     budget_chars: int = 0) -> List[Chunk]:
        """Union of results across sub-queries with three post-processing steps:

        1. Deduplicate by chunk id.
        2. Drop near-identical chunks (Jaccard word-overlap >= 0.8).
        3. If budget_chars > 0, drop lowest-priority chunks once the running
           character total (capped at EVIDENCE_SNIPPET_CHARS per chunk) exceeds
           the budget. Highest-priority chunks (earliest query, best rank) survive.
        """
        # Step 1 — id-based dedup
        seen_ids: set = set()
        candidates: List[Chunk] = []
        for q in queries:
            for c in self.search(q, top_k=per_query):
                if c.id not in seen_ids:
                    seen_ids.add(c.id)
                    candidates.append(c)

        # Step 2 — near-content dedup (Jaccard on word sets)
        kept: List[Chunk] = []
        kept_words: List[frozenset] = []
        for c in candidates:
            words = frozenset(c.text.lower().split())
            if not words:
                kept.append(c)
                continue
            if not any(
                len(words & kw) / max(len(words | kw), 1) >= 0.8
                for kw in kept_words
            ):
                kept.append(c)
                kept_words.append(words)

        # Step 3 — hard character budget (optional)
        if budget_chars <= 0:
            return kept
        snippet_cap = config.EVIDENCE_SNIPPET_CHARS
        selected: List[Chunk] = []
        total = 0
        for c in kept:
            cost = min(len(c.text), snippet_cap) if snippet_cap else len(c.text)
            if total + cost <= budget_chars:
                selected.append(c)
                total += cost
        return selected
