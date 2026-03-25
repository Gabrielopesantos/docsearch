from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docsearch.config import AppConfig
    from docsearch.indexer import IndexStats
    from docsearch.search import RenderedResult


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_COLOR_ENABLED: bool | None = None


def _colors_enabled() -> bool:
    global _COLOR_ENABLED
    if _COLOR_ENABLED is None:
        _COLOR_ENABLED = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    return _COLOR_ENABLED  # type: ignore[return-value]


def _ansi(code: str, text: str) -> str:
    if not _colors_enabled():
        return text
    return f"\033[{code}m{text}\033[0m"


def bold(text: str) -> str:
    return _ansi("1", text)


def dim(text: str) -> str:
    return _ansi("2", text)


def red(text: str) -> str:
    return _ansi("31", text)


def green(text: str) -> str:
    return _ansi("32", text)


def yellow(text: str) -> str:
    return _ansi("33", text)


def cyan(text: str) -> str:
    return _ansi("36", text)


def bold_red(text: str) -> str:
    return _ansi("1;31", text)


def bold_cyan(text: str) -> str:
    return _ansi("1;36", text)


def bold_white(text: str) -> str:
    return _ansi("1;37", text)


# ---------------------------------------------------------------------------
# Score color
# ---------------------------------------------------------------------------


def _score_color(score: float) -> str:
    formatted = f"{score:.4f}"
    if score >= 0.7:
        return green(formatted)
    if score >= 0.4:
        return yellow(formatted)
    return red(formatted)


# ---------------------------------------------------------------------------
# Pretty renderers
# ---------------------------------------------------------------------------


def render_search_results(query: str, results: list[RenderedResult]) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append(f"  {dim('Query:')} {bold_white(query)}")
    lines.append("")

    if not results:
        lines.append(f"  {dim('No matches found.')}")
        lines.append("")
        return "\n".join(lines)

    for i, r in enumerate(results, start=1):
        deleted_tag = f"  {bold_red('DELETED')}" if not r.file_exists else ""
        rank = bold_white(f"{i}.")
        score = f"[{_score_color(r.score)}]"
        file = bold_cyan(r.file_name)
        page = dim(f"p.{r.page_number}")

        lines.append(f"  {rank} {score} {file}  {page}{deleted_tag}")
        lines.append(f"     {dim(r.file_path)}")
        lines.append(f"     {r.snippet}")
        lines.append("")

    return "\n".join(lines)


@dataclass(frozen=True)
class DocumentInfo:
    file_name: str
    file_path: str
    file_hash: str
    page_count: int
    chunk_count: int
    indexed_at: str
    file_exists: bool


def render_status(
    cfg: AppConfig,
    db_path: str,
    db_size_bytes: int,
    total_documents: int,
    total_chunks: int,
    documents: list[DocumentInfo],
) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append(f"  {bold_white('docsearch index status')}")
    lines.append("")

    def _human_size(n: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024:
                return f"{n} {unit}"
            n //= 1024
        return f"{n} TB"

    label_w = 14
    lines.append(
        f"  {'Database':<{label_w}} {dim(db_path)} {dim('(' + _human_size(db_size_bytes) + ')')}"
    )
    lines.append(f"  {'Documents':<{label_w}} {bold_white(str(total_documents))}")
    lines.append(f"  {'Chunks':<{label_w}} {bold_white(str(total_chunks))}")
    lines.append(f"  {'Model':<{label_w}} {cyan(cfg.embedding.model_name)}")
    lines.append(
        f"  {'Chunk size':<{label_w}} {cfg.chunking.chunk_size} chars "
        f"{dim('(' + str(cfg.chunking.chunk_overlap) + ' overlap)')}"
    )
    lines.append(f"  {'Config':<{label_w}} {dim(str(Path('~/.config/docsearch/config.toml')))}")
    lines.append("")

    if documents:
        lines.append(f"  {bold_white('Documents:')}")

        name_w = max(len(d.file_name) for d in documents) + 2
        for d in documents:
            deleted_tag = f"  {bold_red('DELETED')}" if not d.file_exists else ""
            date_part = d.indexed_at[:10] if d.indexed_at else ""
            short_hash = d.file_hash[:12] if d.file_hash else ""
            # Pad the plain name first, then colorize to avoid ANSI codes
            # breaking alignment
            padded_name = d.file_name.ljust(name_w)
            lines.append(
                f"    {bold_cyan(padded_name)}"
                f" {d.page_count:>3} pages"
                f"  {d.chunk_count:>5} chunks"
                f"   {dim(date_part)}"
                f"   {dim(short_hash)}"
                f"{deleted_tag}"
            )
        lines.append("")

    return "\n".join(lines)


def render_index_stats(stats: IndexStats) -> str:
    lines: list[str] = []
    lines.append("")
    lines.append(f"  {bold_white('Indexing complete')}")
    lines.append("")
    label_w = 12
    lines.append(f"  {'Scanned':<{label_w}} {stats.scanned}")
    lines.append(f"  {'Indexed':<{label_w}} {green(str(stats.indexed))}")
    if stats.reindexed > 0:
        lines.append(f"  {'Re-indexed':<{label_w}} {yellow(str(stats.reindexed))}")
    lines.append(f"  {'Skipped':<{label_w}} {dim(str(stats.skipped))}")
    lines.append(f"  {'Chunks':<{label_w}} {stats.chunks}")
    lines.append("")
    return "\n".join(lines)


def render_sync(removed_docs: int, removed_chunks: int) -> str:
    lines: list[str] = []
    lines.append("")
    if removed_docs == 0:
        lines.append(f"  {green('All indexed files still exist on disk. Nothing to remove.')}")
    else:
        lines.append(
            f"  Removed {bold_red(str(removed_docs))} document(s) "
            f"({bold_red(str(removed_chunks))} chunks) for files no longer on disk."
        )
    lines.append("")
    return "\n".join(lines)


def render_remove(removed: int) -> str:
    lines: list[str] = []
    lines.append("")
    if removed == 0:
        lines.append(f"  {yellow('No matching document found.')}")
    else:
        lines.append(f"  Removed {bold_white(str(removed))} document(s).")
    lines.append("")
    return "\n".join(lines)
