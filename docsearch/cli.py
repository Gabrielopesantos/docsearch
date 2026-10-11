from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from docsearch.config import DEFAULT_CONFIG_PATH, load_config, with_overrides


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="docsearch",
        description="Index and search PDFs with local embeddings",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config TOML")
    parser.add_argument("--db", help="Override database path")
    parser.add_argument("--model", help="Override embedding model name")
    parser.add_argument("--batch-size", type=int, help="Override embedding batch size")
    parser.add_argument("--chunk-size", type=int, help="Override chunk size")
    parser.add_argument("--chunk-overlap", type=int, help="Override chunk overlap")
    parser.add_argument(
        "-o",
        "--output",
        choices=["pretty", "json"],
        default="pretty",
        help="Output format (default: pretty)",
    )
    parser.add_argument(
        "--update-model",
        action="store_true",
        help="Re-download the embedding model from Hugging Face Hub",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Index all PDFs in a directory or a single PDF file")
    p_index.add_argument("path", help="Directory containing PDFs or a single PDF file")
    p_index.add_argument(
        "--reindex",
        action="store_true",
        help="Re-index existing paths when content changed",
    )

    p_search = sub.add_parser("search", help="Search indexed PDF content")
    p_search.add_argument("query", help="Natural language query")
    p_search.add_argument("-n", "--top-k", type=int, help="Number of results")
    p_search.add_argument(
        "--score-threshold",
        type=float,
        help="Filter by minimum similarity score",
    )
    p_search.add_argument(
        "--doc",
        dest="docs",
        metavar="PATH_OR_HASH",
        action="append",
        default=None,
        help="Limit search to this document (repeat for multiple); accepts file path or SHA-256 hash",
    )

    p_eval = sub.add_parser("eval", help="Score search quality against an eval file")
    p_eval.add_argument("eval_file", help="TOML file with [[case]] query/expect entries")

    sub.add_parser("status", help="Show index stats and document listing")
    sub.add_parser("sync", help="Remove index entries for PDFs that no longer exist on disk")

    p_remove = sub.add_parser("remove", help="Remove a document by path or hash")
    p_remove.add_argument("key", help="Absolute file path or SHA-256 hash")

    return parser


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _cmd_index(args: argparse.Namespace, cfg, db, embedder, output_fmt: str) -> None:
    from docsearch.indexer import index_path

    stats = index_path(
        path=args.path,
        db=db,
        cfg=cfg,
        embedder=embedder,
        reindex=args.reindex,
    )

    if output_fmt == "json":
        print(json.dumps(asdict(stats), indent=2))
    else:
        from docsearch.display import render_index_stats

        print(render_index_stats(stats))


def _cmd_search(args: argparse.Namespace, cfg, db, embedder, output_fmt: str) -> None:
    import sys

    from docsearch.search import search_query

    try:
        results = search_query(
            query=args.query,
            db=db,
            cfg=cfg,
            embedder=embedder,
            top_k=args.top_k,
            score_threshold=args.score_threshold,
            document_keys=args.docs,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    if output_fmt == "json":
        print(
            json.dumps(
                {
                    "query": args.query,
                    "results": [
                        {
                            "rank": i,
                            "score": round(r.score, 4),
                            "file_name": r.file_name,
                            "file_path": r.file_path,
                            "file_exists": r.file_exists,
                            "page": r.page_number,
                            "chunk_index": r.chunk_index,
                            "snippet": r.snippet,
                        }
                        for i, r in enumerate(results, start=1)
                    ],
                },
                indent=2,
            )
        )
    else:
        from docsearch.display import render_search_results

        print(render_search_results(args.query, results))


def _cmd_eval(args: argparse.Namespace, cfg, db, embedder, output_fmt: str) -> None:
    import sys

    from docsearch.evaluate import load_cases, run_eval

    try:
        cases = load_cases(Path(args.eval_file))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    report = run_eval(cases, db=db, cfg=cfg, embedder=embedder)

    if output_fmt == "json":
        print(
            json.dumps(
                {
                    "model": cfg.embedding.model_name,
                    "cases": len(report.cases),
                    "recall_at_1": round(report.recall_at_1, 4),
                    "recall_at_5": round(report.recall_at_5, 4),
                    "mrr": round(report.mrr, 4),
                    "results": [
                        {
                            "query": c.case.query,
                            "expect": {"file": c.case.file, "page": c.case.page},
                            "rank": c.rank,
                            "top": [{"file_name": f, "page": p} for f, p in c.top],
                        }
                        for c in report.cases
                    ],
                },
                indent=2,
            )
        )
    else:
        from docsearch.display import render_eval

        print(render_eval(report))


def _cmd_status(cfg, db, output_fmt: str) -> None:
    from docsearch.display import DocumentInfo, render_status

    db_path = cfg.storage.db_path
    try:
        db_size = Path(db_path).stat().st_size
    except OSError:
        db_size = 0

    totals = db.stats()
    rows = db.list_documents_with_chunk_counts()

    documents = [
        DocumentInfo(
            file_name=r.file_name,
            file_path=r.file_path,
            file_hash=r.file_hash,
            page_count=r.page_count,
            chunk_count=r.chunk_count,
            indexed_at=r.indexed_at,
            file_exists=Path(r.file_path).exists(),
        )
        for r in rows
    ]

    if output_fmt == "json":
        print(
            json.dumps(
                {
                    "db_path": db_path,
                    "db_size_bytes": db_size,
                    "total_documents": totals["documents"],
                    "total_chunks": totals["chunks"],
                    "model": cfg.embedding.model_name,
                    "chunk_size": cfg.chunking.chunk_size,
                    "chunk_overlap": cfg.chunking.chunk_overlap,
                    "documents": [
                        {
                            "file_name": d.file_name,
                            "file_path": d.file_path,
                            "file_hash": d.file_hash,
                            "file_exists": d.file_exists,
                            "page_count": d.page_count,
                            "chunk_count": d.chunk_count,
                            "indexed_at": d.indexed_at,
                        }
                        for d in documents
                    ],
                },
                indent=2,
            )
        )
    else:
        print(
            render_status(
                cfg=cfg,
                db_path=db_path,
                db_size_bytes=db_size,
                total_documents=totals["documents"],
                total_chunks=totals["chunks"],
                documents=documents,
            )
        )


def _cmd_sync(db, output_fmt: str) -> None:
    from docsearch.display import render_sync

    all_docs = db.list_all_documents()
    missing = [d for d in all_docs if not Path(d.file_path).exists()]

    if not missing:
        if output_fmt == "json":
            print(json.dumps({"removed_documents": 0, "removed_chunks": 0}, indent=2))
        else:
            print(render_sync(0, 0))
        return

    missing_ids = [d.id for d in missing]
    chunk_count = db.count_chunks_for_documents(missing_ids)
    db.delete_documents_batch(missing_ids)

    if output_fmt == "json":
        print(
            json.dumps(
                {
                    "removed_documents": len(missing),
                    "removed_chunks": chunk_count,
                },
                indent=2,
            )
        )
    else:
        print(render_sync(len(missing), chunk_count))


def _cmd_remove(args: argparse.Namespace, db, output_fmt: str) -> None:
    removed = db.delete_document_by_path_or_hash(args.key)

    if output_fmt == "json":
        print(json.dumps({"removed": removed}, indent=2))
    else:
        from docsearch.display import render_remove

        print(render_remove(removed))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg, created = load_config(Path(args.config))
    cfg = with_overrides(
        cfg,
        db_path=args.db,
        model_name=args.model,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=getattr(args, "top_k", None),
        score_threshold=getattr(args, "score_threshold", None),
        score_threshold_set=getattr(args, "score_threshold", None) is not None,
    )

    output_fmt = args.output
    update_model = args.update_model

    from docsearch.database import Database

    embedder = None
    vector_dim = 384  # default; overridden below for commands that load the model
    if args.command in ("index", "search", "eval"):
        from docsearch.embedding import load_embedder

        embedder = load_embedder(
            model_name=cfg.embedding.model_name,
            cache_dir=cfg.embedding.model_cache_dir,
            update=update_model,
        )
        vector_dim = embedder.dim

    db = Database(cfg.storage.db_path, vector_dim=vector_dim)

    try:
        if created and output_fmt == "pretty":
            from docsearch.display import dim

            print(dim(f"Created default config at {Path(args.config).expanduser()}"))

        if args.command == "index":
            _cmd_index(args, cfg, db, embedder, output_fmt)
        elif args.command == "search":
            _cmd_search(args, cfg, db, embedder, output_fmt)
        elif args.command == "eval":
            _cmd_eval(args, cfg, db, embedder, output_fmt)
        elif args.command == "status":
            _cmd_status(cfg, db, output_fmt)
        elif args.command == "sync":
            _cmd_sync(db, output_fmt)
        elif args.command == "remove":
            _cmd_remove(args, db, output_fmt)
    finally:
        db.close()


if __name__ == "__main__":
    main()
