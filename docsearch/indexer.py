from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import pymupdf

from docsearch.chunking import chunk_text
from docsearch.config import AppConfig
from docsearch.database import Database
from docsearch.embedding import Embedder

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
    no_text: int = 0
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


# Header/footer candidates: this many non-empty lines at the top and bottom of a page.
_EDGE_LINES = 3


def _normalize_line(line: str) -> str:
    # Digits vary per page ("page 3" vs "page 4"), so compare lines without them.
    return re.sub(r"\d+", "#", " ".join(line.split())).lower()


def strip_repeated_edges(pages: list[str]) -> list[str]:
    """Remove header/footer lines that repeat on more than half of the pages."""
    if len(pages) < 3:
        return pages
    page_lines = [p.splitlines() for p in pages]
    edges_per_page: list[set[int]] = []
    counts: Counter[str] = Counter()
    for lines in page_lines:
        nonempty = [i for i, line in enumerate(lines) if line.strip()]
        edges = set(nonempty[:_EDGE_LINES] + nonempty[-_EDGE_LINES:])
        edges_per_page.append(edges)
        counts.update({_normalize_line(lines[i]) for i in edges})
    repeated = {line for line, n in counts.items() if n > len(pages) / 2}
    return [
        "\n".join(
            line
            for i, line in enumerate(lines)
            if not (i in edges and _normalize_line(line) in repeated)
        )
        for lines, edges in zip(page_lines, edges_per_page, strict=True)
    ]


def extract_page_chunks(
    path: Path,
    max_tokens: int,
    overlap_sentences: int,
    count_tokens: Callable[[str], int],
) -> tuple[list[PageChunk], int, str]:
    """Return (chunks, page count, document title)."""
    doc = pymupdf.open(path)
    try:
        page_count = doc.page_count
        title = (doc.metadata or {}).get("title", "").strip() or path.stem
        pages = strip_repeated_edges([page.get_text("text") for page in doc])
    finally:
        doc.close()

    page_chunks = [
        PageChunk(page_number=i, text=chunk)
        for i, text in enumerate(pages, start=1)
        for chunk in chunk_text(
            text,
            max_tokens=max_tokens,
            overlap_sentences=overlap_sentences,
            count_tokens=count_tokens,
        )
    ]
    return page_chunks, page_count, title


_PDF_MAGIC = b"%PDF-"


def _is_pdf_file(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(5) == _PDF_MAGIC
    except OSError:
        return False


def _index_file(
    pdf_path: Path,
    db: Database,
    cfg: AppConfig,
    embedder: Embedder,
    reindex: bool,
    stats: IndexStats,
) -> tuple[IndexStats, str]:
    """Index one PDF. Returns updated stats and a short note for progress output."""
    replaced = False
    file_hash = sha256_file(pdf_path)
    for existing in (db.get_document_by_hash(file_hash), db.get_document_by_path(str(pdf_path))):
        if existing is None:
            continue
        if not reindex:
            return replace(stats, skipped=stats.skipped + 1), "skipped (already indexed)"
        db.delete_document(existing.id)
        replaced = True

    page_chunks, page_count, title = extract_page_chunks(
        pdf_path,
        max_tokens=cfg.chunking.max_tokens,
        overlap_sentences=cfg.chunking.overlap_sentences,
        count_tokens=embedder.count_tokens,
    )
    if not page_chunks:
        # Record it so the next run skips it, but report it apart.
        db.insert_document(
            file_path=str(pdf_path),
            file_hash=file_hash,
            file_name=pdf_path.name,
            page_count=page_count,
        )
        return replace(stats, no_text=stats.no_text + 1), "no text (scanned? run ocrmypdf first)"

    # The title gives each chunk document context; the stored text stays raw.
    texts = [f"{title}\n{c.text}" for c in page_chunks]
    vector_rows = embedder.encode_passages(texts, batch_size=cfg.embedding.batch_size)

    doc_id = db.insert_document(
        file_path=str(pdf_path),
        file_hash=file_hash,
        file_name=pdf_path.name,
        page_count=page_count,
    )
    rows = [(i, c.page_number, c.text, vector_rows[i]) for i, c in enumerate(page_chunks)]
    inserted = db.insert_chunks(doc_id, rows)

    stats = replace(stats, chunks=stats.chunks + inserted)
    if replaced:
        return replace(stats, reindexed=stats.reindexed + 1), f"re-indexed, {inserted} chunks"
    return replace(stats, indexed=stats.indexed + 1), f"{inserted} chunks"


def index_path(
    path: str,
    db: Database,
    cfg: AppConfig,
    embedder: Embedder,
    reindex: bool = False,
    progress: Callable[[str], None] = logger.info,
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
    stats = IndexStats(scanned=len(pdfs))

    for i, pdf_path in enumerate(pdfs, start=1):
        started = time.monotonic()
        try:
            stats, note = _index_file(pdf_path, db, cfg, embedder, reindex, stats)
        except Exception as exc:
            stats, note = replace(stats, skipped=stats.skipped + 1), f"failed: {exc}"
        elapsed = time.monotonic() - started
        progress(f"[{i}/{len(pdfs)}] {pdf_path.name}: {note} ({elapsed:.1f}s)")

    return stats
