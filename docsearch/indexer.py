from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import pymupdf

from docsearch.chunking import chunk_text
from docsearch.config import AppConfig
from docsearch.database import Database
from docsearch.embedding import load_embedder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageChunk:
    page_number: int
    text: str


@dataclass(frozen=True)
class IndexStats:
    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    reindexed: int = 0
    chunks: int = 0


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def extract_page_chunks(
    path: Path, chunk_size: int, chunk_overlap: int
) -> tuple[list[PageChunk], int]:
    doc = pymupdf.open(path)
    page_chunks: list[PageChunk] = []
    page_count = doc.page_count
    try:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            page_chunks.extend(
                PageChunk(page_number=i, text=chunk)
                for chunk in chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            )
    finally:
        doc.close()
    return page_chunks, page_count


_PDF_MAGIC = b"%PDF-"


def _is_pdf_file(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(5) == _PDF_MAGIC
    except OSError:
        return False


def index_path(
    path: str,
    db: Database,
    cfg: AppConfig,
    reindex: bool = False,
    update_model: bool = False,
) -> IndexStats:
    base = Path(path).expanduser().resolve()
    if base.is_file():
        if not _is_pdf_file(base):
            raise ValueError(f"Not a PDF file: {base}")
        pdfs = [base]
    elif base.is_dir():
        pdfs = sorted(base.rglob("*.pdf"))
    else:
        raise ValueError(f"Path does not exist: {base}")
    model = load_embedder(
        model_name=cfg.embedding.model_name,
        cache_dir=cfg.embedding.model_cache_dir,
        update=update_model,
    )

    stats = IndexStats(scanned=len(pdfs))

    for pdf_path in pdfs:
        try:
            replaced = False
            file_hash = sha256_file(pdf_path)
            existing_hash = db.get_document_by_hash(file_hash)
            if existing_hash is not None:
                if reindex:
                    db.delete_document(existing_hash.id)
                    replaced = True
                else:
                    stats = replace(stats, skipped=stats.skipped + 1)
                    continue

            existing_path = db.get_document_by_path(str(pdf_path))
            if existing_path is not None:
                if reindex:
                    db.delete_document(existing_path.id)
                    replaced = True
                else:
                    stats = replace(stats, skipped=stats.skipped + 1)
                    continue

            page_chunks, page_count = extract_page_chunks(
                pdf_path,
                chunk_size=cfg.chunking.chunk_size,
                chunk_overlap=cfg.chunking.chunk_overlap,
            )
            if not page_chunks:
                doc_id = db.insert_document(
                    file_path=str(pdf_path),
                    file_hash=file_hash,
                    file_name=pdf_path.name,
                    page_count=page_count,
                )
                if doc_id:
                    if replaced:
                        stats = replace(stats, reindexed=stats.reindexed + 1)
                    else:
                        stats = replace(stats, indexed=stats.indexed + 1)
                continue

            texts = [c.text for c in page_chunks]
            vectors = model.encode(
                texts,
                batch_size=cfg.embedding.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vector_rows = vectors.tolist()

            doc_id = db.insert_document(
                file_path=str(pdf_path),
                file_hash=file_hash,
                file_name=pdf_path.name,
                page_count=page_count,
            )

            rows = [(i, c.page_number, c.text, vector_rows[i]) for i, c in enumerate(page_chunks)]
            inserted_chunks = db.insert_chunks(doc_id, rows)

            if replaced:
                stats = replace(stats, reindexed=stats.reindexed + 1, chunks=stats.chunks + inserted_chunks)
            else:
                stats = replace(stats, indexed=stats.indexed + 1, chunks=stats.chunks + inserted_chunks)

        except Exception as exc:
            logger.warning("Skipping %s: %s", pdf_path, exc)
            stats = replace(stats, skipped=stats.skipped + 1)

    return stats
