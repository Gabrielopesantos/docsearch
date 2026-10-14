from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from docsearch.config import AppConfig
from docsearch.database import Database
from docsearch.display import RenderedResult
from docsearch.embedding import Embedder
from docsearch.rerank import Reranker
from docsearch.search import search_query

# Results fetched per query. MRR counts a hit only inside this window.
FETCH_K = 10


@dataclass(frozen=True)
class EvalCase:
    query: str
    file: str
    page: int | None


@dataclass(frozen=True)
class CaseResult:
    case: EvalCase
    rank: int | None
    top: list[tuple[str, int]]


@dataclass(frozen=True)
class EvalReport:
    cases: list[CaseResult]
    recall_at_1: float
    recall_at_5: float
    mrr: float


def load_cases(path: Path) -> list[EvalCase]:
    try:
        with path.expanduser().open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid TOML in eval file {path}: {exc}") from exc

    cases: list[EvalCase] = []
    for i, raw in enumerate(data.get("case", []), start=1):
        query = raw.get("query")
        expect = raw.get("expect", {})
        file = expect.get("file")
        if not query or not file:
            raise ValueError(f"case #{i}: 'query' and 'expect.file' are required")
        page = expect.get("page")
        cases.append(EvalCase(query=query, file=file, page=None if page is None else int(page)))
    if not cases:
        raise ValueError(f"no [[case]] entries in {path}")
    return cases


def _is_hit(case: EvalCase, result: RenderedResult) -> bool:
    if result.file_name != case.file and Path(result.file_path).name != case.file:
        return False
    return case.page is None or result.page_number == case.page


def run_eval(
    cases: list[EvalCase],
    db: Database,
    cfg: AppConfig,
    embedder: Embedder,
    reranker: Reranker | None = None,
) -> EvalReport:
    out: list[CaseResult] = []
    for case in cases:
        results = search_query(
            query=case.query,
            db=db,
            cfg=cfg,
            embedder=embedder,
            reranker=reranker,
            top_k=FETCH_K,
            score_threshold=None,
        )
        rank = next((i for i, r in enumerate(results, start=1) if _is_hit(case, r)), None)
        out.append(
            CaseResult(
                case=case,
                rank=rank,
                top=[(r.file_name, r.page_number) for r in results[:5]],
            )
        )

    n = len(out)
    return EvalReport(
        cases=out,
        recall_at_1=sum(1 for c in out if c.rank == 1) / n,
        recall_at_5=sum(1 for c in out if c.rank is not None and c.rank <= 5) / n,
        mrr=sum(1 / c.rank for c in out if c.rank is not None) / n,
    )
