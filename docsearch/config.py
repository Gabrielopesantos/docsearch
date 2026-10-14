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
    model_name: str = "intfloat/multilingual-e5-small"
    model_cache_dir: str = str(DEFAULT_MODEL_CACHE_DIR)
    batch_size: int = 64
    device: str = "auto"


@dataclass(frozen=True)
class ChunkingConfig:
    max_tokens: int = 250
    overlap_sentences: int = 1


@dataclass(frozen=True)
class SearchConfig:
    top_k: int = 5
    score_threshold: float | None = None


@dataclass(frozen=True)
class RerankConfig:
    enabled: bool = True
    model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


@dataclass(frozen=True)
class StorageConfig:
    db_path: str = str(DEFAULT_DB_PATH)


@dataclass(frozen=True)
class AppConfig:
    embedding: EmbeddingConfig = EmbeddingConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    search: SearchConfig = SearchConfig()
    rerank: RerankConfig = RerankConfig()
    storage: StorageConfig = StorageConfig()


DEFAULT_CONFIG = AppConfig()

DEFAULT_CONFIG_TEMPLATE = """# docsearch configuration file
# Precedence: CLI flags > this file > built-in defaults

[embedding]
# Local sentence-transformers model name. The index remembers which model
# built it; changing this needs a new index (--db or delete the index file).
# Multilingual and CPU-friendly default. With a GPU, "BAAI/bge-m3" is stronger.
model_name = "intfloat/multilingual-e5-small"
# Where to cache downloaded model files
model_cache_dir = "~/.cache/docsearch/models"
# Number of chunks embedded per batch
batch_size = 64
# "auto" (GPU if available), "cpu" or "cuda". Changes speed only, not results.
device = "auto"

[chunking]
# Maximum tokens per chunk. Chunks hold whole sentences and never cross a page.
# Keep below the model's input limit (512 for e5-small).
max_tokens = 250
# Sentences repeated from the previous chunk
overlap_sentences = 1

[search]
# Number of results returned by default
top_k = 5
# Optional minimum score in [0,1], or omit. With rerank enabled this is the
# reranker score; without it, the fused rank score.
# score_threshold = 0.2

[rerank]
# Cross-encoder that re-scores the top candidates. Scores are 0..1.
# Multilingual and CPU-friendly default. With a GPU, "BAAI/bge-reranker-v2-m3" is stronger.
enabled = true
model_name = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

[storage]
# Local libSQL database file path
db_path = "~/.docsearch/index.db"
"""


def _expand_path(value: str) -> str:
    return str(Path(value).expanduser())


_DEVICES = ("auto", "cpu", "cuda")


def _check_device(value: str) -> str:
    if value not in _DEVICES:
        raise SystemExit(f"error: device must be one of {', '.join(_DEVICES)}, got {value!r}")
    return value


def _check_max_tokens(value: int) -> int:
    if value <= 0:
        raise SystemExit(f"error: max_tokens must be > 0, got {value}")
    return value


def _check_overlap(value: int) -> int:
    if value < 0:
        raise SystemExit(f"error: overlap_sentences must be >= 0, got {value}")
    return value


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
    try:
        with config_path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise SystemExit(f"error: invalid TOML in config file {config_path}: {exc}") from exc

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
                device=_check_device(str(embedding.get("device", cfg.embedding.device))),
            ),
        )

    chunking = data.get("chunking", {})
    if chunking:
        cfg = replace(
            cfg,
            chunking=replace(
                cfg.chunking,
                max_tokens=_check_max_tokens(
                    int(chunking.get("max_tokens", cfg.chunking.max_tokens))
                ),
                overlap_sentences=_check_overlap(
                    int(chunking.get("overlap_sentences", cfg.chunking.overlap_sentences))
                ),
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

    rerank = data.get("rerank", {})
    if rerank:
        cfg = replace(
            cfg,
            rerank=replace(
                cfg.rerank,
                enabled=bool(rerank.get("enabled", cfg.rerank.enabled)),
                model_name=rerank.get("model_name", cfg.rerank.model_name),
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
    device: str | None = None,
    max_tokens: int | None = None,
    overlap_sentences: int | None = None,
    top_k: int | None = None,
    score_threshold: float | None = None,
    score_threshold_set: bool = False,
    rerank: bool | None = None,
) -> AppConfig:
    out = cfg
    if db_path is not None:
        out = replace(out, storage=replace(out.storage, db_path=_expand_path(db_path)))
    if model_name is not None:
        out = replace(out, embedding=replace(out.embedding, model_name=model_name))
    if batch_size is not None:
        out = replace(out, embedding=replace(out.embedding, batch_size=batch_size))
    if device is not None:
        out = replace(out, embedding=replace(out.embedding, device=_check_device(device)))
    if max_tokens is not None:
        out = replace(out, chunking=replace(out.chunking, max_tokens=_check_max_tokens(max_tokens)))
    if overlap_sentences is not None:
        out = replace(
            out,
            chunking=replace(out.chunking, overlap_sentences=_check_overlap(overlap_sentences)),
        )
    if top_k is not None:
        out = replace(out, search=replace(out.search, top_k=top_k))
    if rerank is not None:
        out = replace(out, rerank=replace(out.rerank, enabled=rerank))
    if score_threshold_set:
        out = replace(out, search=replace(out.search, score_threshold=score_threshold))
    return out
