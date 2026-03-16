from __future__ import annotations

from sentence_transformers import SentenceTransformer


def load_embedder(
    model_name: str,
    cache_dir: str,
    update: bool = False,
) -> SentenceTransformer:
    """Load a sentence-transformers model.

    Default behaviour is local-first: if the model is already cached no
    network requests are made.  On the very first run the model is
    downloaded automatically.  Pass ``update=True`` to force a fresh
    download from Hugging Face Hub (e.g. to pick up upstream weight
    updates).
    """
    if update:
        return SentenceTransformer(model_name, cache_folder=cache_dir)

    try:
        return SentenceTransformer(
            model_name,
            cache_folder=cache_dir,
            local_files_only=True,
        )
    except OSError:
        # First run -- model not cached yet, must download
        return SentenceTransformer(model_name, cache_folder=cache_dir)
