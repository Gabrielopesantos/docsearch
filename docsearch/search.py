from __future__ import annotations

import re
from pathlib import Path

from docsearch.config import AppConfig
from docsearch.database import Database
from docsearch.display import RenderedResult
from docsearch.embedding import Embedder

# Candidates taken from each retriever before fusion.
CANDIDATES = 50
# Reciprocal rank fusion constant; 60 is the value from the original RRF paper.
RRF_K = 60

_WORD = re.compile(r"\w+")


def fts_query(query: str) -> str | None:
    """Turn free text into a safe FTS5 expression.

    Each whitespace-separated term becomes a quoted phrase of its word parts,
    so "A.8.2" matches the adjacent tokens "a 8 2" and FTS5 operators in user
    input are never interpreted. Terms are joined with OR; bm25 ranks chunks
    that match more and rarer terms higher.
    """
    phrases = []
    for term in query.split():
        parts = _WORD.findall(term)
        if parts:
            phrases.append('"' + " ".join(parts) + '"')
    return " OR ".join(phrases) if phrases else None


def rrf_fuse(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Merge ranked id lists. Returns (id, score) best first, score scaled to 0..1."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    best_possible = len(rankings) / (k + 1)
    return sorted(
        ((item, score / best_possible) for item, score in scores.items()),
        key=lambda pair: pair[1],
        reverse=True,
    )


def _truncate(text: str, limit: int = 220) -> str:
    value = " ".join(text.split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def search_query(
    query: str,
    db: Database,
    cfg: AppConfig,
    embedder: Embedder,
    top_k: int | None = None,
    score_threshold: float | None = None,
    document_keys: list[str] | None = None,
) -> list[RenderedResult]:
    requested_k = top_k if top_k is not None else cfg.search.top_k
    threshold = score_threshold if score_threshold is not None else cfg.search.score_threshold
    pool = max(CANDIDATES, requested_k)

    doc_ids = db.resolve_document_ids(document_keys) if document_keys else None

    vector_ids = db.vector_candidates(embedder.encode_query(query), pool, document_ids=doc_ids)
    match_expr = fts_query(query)
    keyword_ids = (
        db.keyword_candidates(match_expr, pool, document_ids=doc_ids) if match_expr else []
    )

    fused = rrf_fuse([vector_ids, keyword_ids])[:requested_k]
    chunks = db.get_chunks([chunk_id for chunk_id, _ in fused])

    out: list[RenderedResult] = []
    for chunk_id, score in fused:
        if threshold is not None and score < threshold:
            continue
        row = chunks[chunk_id]
        out.append(
            RenderedResult(
                file_name=row.file_name,
                file_path=row.file_path,
                page_number=row.page_number,
                chunk_index=row.chunk_index,
                score=score,
                snippet=_truncate(row.content),
                file_exists=Path(row.file_path).exists(),
            )
        )
    return out
