from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "docsearch" / "config.toml"
DEFAULT_DB_PATH = Path.home() / ".docsearch" / "index.db"
DEFAULT_MODEL_CACHE_DIR = Path.home() / ".cache" / "docsearch" / "models"


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    model_cache_dir: str = str(DEFAULT_MODEL_CACHE_DIR)
    batch_size: int = 64


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 500
    chunk_overlap: int = 100


@dataclass(frozen=True)
class SearchConfig:
    top_k: int = 5
    score_threshold: float | None = None


@dataclass(frozen=True)
class StorageConfig:
    db_path: str = str(DEFAULT_DB_PATH)


@dataclass(frozen=True)
class AppConfig:
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    search: SearchConfig = SearchConfig()
    storage: StorageConfig = StorageConfig()


DEFAULT_CONFIG = AppConfig()

DEFAULT_CONFIG_TEMPLATE = """# docsearch configuration file
# Precedence: CLI flags > this file > built-in defaults

[embedding]
# Local sentence-transformers model name
model_name = "all-MiniLM-L6-v2"
# Where to cache downloaded model files
model_cache_dir = "~/.cache/docsearch/models"
# Number of chunks embedded per batch
batch_size = 64

[chunking]
# Characters per chunk
chunk_size = 500
# Overlap characters between chunks
chunk_overlap = 100

[search]
# Number of results returned by default
top_k = 5
# Optional cosine similarity threshold in [0,1], or omit
# score_threshold = 0.35

[storage]
# Local libSQL database file path
db_path = "~/.docsearch/index.db"
"""


def _expand_path(value: str) -> str:
    return str(Path(value).expanduser())


def ensure_default_config(config_path: Path = DEFAULT_CONFIG_PATH) -> tuple[Path, bool]:
    config_path = config_path.expanduser()
    if config_path.exists():
        return config_path, False
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(DEFAULT_CONFIG_TEMPLATE, encoding="utf-8")
    return config_path, True


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> tuple[AppConfig, bool]:
    config_path, created = ensure_default_config(config_path)
    data: dict[str, Any]
    with config_path.open("rb") as f:
        data = tomllib.load(f)

    cfg = DEFAULT_CONFIG

    embedding = data.get("embedding", {})
    if embedding:
        cfg = replace(
            cfg,
            embedding=replace(
                cfg.embedding,
                model_name=embedding.get("model_name", cfg.embedding.model_name),
                model_cache_dir=_expand_path(
                    embedding.get("model_cache_dir", cfg.embedding.model_cache_dir)
                ),
                batch_size=int(embedding.get("batch_size", cfg.embedding.batch_size)),
            ),
        )

    chunking = data.get("chunking", {})
    if chunking:
        cfg = replace(
            cfg,
            chunking=replace(
                cfg.chunking,
                chunk_size=int(chunking.get("chunk_size", cfg.chunking.chunk_size)),
                chunk_overlap=int(chunking.get("chunk_overlap", cfg.chunking.chunk_overlap)),
            ),
        )

    search = data.get("search", {})
    if search:
        raw_threshold = search.get("score_threshold", cfg.search.score_threshold)
        threshold = None if raw_threshold is None else float(raw_threshold)
        cfg = replace(
            cfg,
            search=replace(
                cfg.search,
                top_k=int(search.get("top_k", cfg.search.top_k)),
                score_threshold=threshold,
            ),
        )

    storage = data.get("storage", {})
    if storage:
        cfg = replace(
            cfg,
            storage=replace(
                cfg.storage,
                db_path=_expand_path(storage.get("db_path", cfg.storage.db_path)),
            ),
        )

    return cfg, created


def with_overrides(
    cfg: AppConfig,
    db_path: str | None = None,
    model_name: str | None = None,
    batch_size: int | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    top_k: int | None = None,
    score_threshold: float | None = None,
    score_threshold_set: bool = False,
) -> AppConfig:
    out = cfg
    if db_path is not None:
        out = replace(out, storage=replace(out.storage, db_path=_expand_path(db_path)))
    if model_name is not None:
        out = replace(out, embedding=replace(out.embedding, model_name=model_name))
    if batch_size is not None:
        out = replace(out, embedding=replace(out.embedding, batch_size=batch_size))
    if chunk_size is not None:
        out = replace(out, chunking=replace(out.chunking, chunk_size=chunk_size))
    if chunk_overlap is not None:
        out = replace(out, chunking=replace(out.chunking, chunk_overlap=chunk_overlap))
    if top_k is not None:
        out = replace(out, search=replace(out.search, top_k=top_k))
    if score_threshold_set:
        out = replace(out, search=replace(out.search, score_threshold=score_threshold))
    return out
