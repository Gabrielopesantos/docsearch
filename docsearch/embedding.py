from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from transformers.utils import logging as hf_logging


def default_prefixes(model_name: str) -> tuple[str, str]:
    """Return (query_prefix, passage_prefix) the model was trained with.

    The e5 family is trained with "query: " / "passage: " markers and loses
    accuracy without them. Most other models (MiniLM, bge-m3) need none.
    """
    if "e5" in model_name.lower().split("/")[-1]:
        return "query: ", "passage: "
    return "", ""


@dataclass(frozen=True)
class Embedder:
    model: SentenceTransformer
    model_name: str
    query_prefix: str
    passage_prefix: str

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.tokenize(text))

    def encode_query(self, query: str) -> list[float]:
        return self.model.encode(
            [self.query_prefix + query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()

    def encode_passages(self, texts: list[str], batch_size: int) -> list[list[float]]:
        return self.model.encode(
            [self.passage_prefix + t for t in texts],
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()


def quiet_transformers() -> None:
    """Hide transformers load reports, tokenizer length warnings and weight-loading bars.

    Hub download progress still shows, so a first-run download is visible.
    """
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()


def resolve_device(device: str) -> str | None:
    """Map the config value to a sentence-transformers device ("auto" lets it pick)."""
    return None if device == "auto" else device


def load_embedder(
    model_name: str,
    cache_dir: str,
    update: bool = False,
    device: str = "auto",
) -> Embedder:
    """Load a sentence-transformers model.

    Default behaviour is local-first: if the model is already cached no
    network requests are made.  On the very first run the model is
    downloaded automatically.  Pass ``update=True`` to force a fresh
    download from Hugging Face Hub (e.g. to pick up upstream weight
    updates).
    """
    quiet_transformers()
    dev = resolve_device(device)
    if update:
        model = SentenceTransformer(model_name, cache_folder=cache_dir, device=dev)
    else:
        try:
            model = SentenceTransformer(
                model_name,
                cache_folder=cache_dir,
                local_files_only=True,
                device=dev,
            )
        except OSError:
            # First run -- model not cached yet, must download
            model = SentenceTransformer(model_name, cache_folder=cache_dir, device=dev)

    query_prefix, passage_prefix = default_prefixes(model_name)
    return Embedder(
        model=model,
        model_name=model_name,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
    )
