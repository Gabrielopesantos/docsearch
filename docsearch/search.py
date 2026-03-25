from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from docsearch.config import AppConfig
from docsearch.database import Database
from docsearch.embedding import load_embedder


@dataclass(frozen=True)
class RenderedResult:
    file_name: str
    file_path: str
    page_number: int
    chunk_index: int
    score: float
    snippet: str
    file_exists: bool


def _to_similarity(cosine_distance: float) -> float:
    return 1.0 - cosine_distance


def _truncate(text: str, limit: int = 220) -> str:
    value = " ".join(text.split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def search_query(
    query: str,
    db: Database,
    cfg: AppConfig,
    top_k: int | None = None,
    score_threshold: float | None = None,
    update_model: bool = False,
    document_keys: list[str] | None = None,
) -> list[RenderedResult]:
    requested_k = top_k if top_k is not None else cfg.search.top_k
    threshold = score_threshold if score_threshold is not None else cfg.search.score_threshold

    model = load_embedder(
        model_name=cfg.embedding.model_name,
        cache_dir=cfg.embedding.model_cache_dir,
        update=update_model,
    )
    vector = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()

    doc_ids = db.resolve_document_ids(document_keys) if document_keys else None
    rows = db.search_chunks(vector, requested_k, document_ids=doc_ids)
    out: list[RenderedResult] = []
    for row in rows:
        score = _to_similarity(row.cosine_distance)
        if threshold is not None and score < threshold:
            continue
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
