from __future__ import annotations

from dataclasses import dataclass

import torch
from sentence_transformers import CrossEncoder

from docsearch.embedding import quiet_transformers, resolve_device


@dataclass(frozen=True)
class Reranker:
    """Cross-encoder that reads query and passage together and scores relevance.

    Slower than vector search, so it only re-orders the fused candidates.
    """

    model: CrossEncoder

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        # Sigmoid maps the raw logits to 0..1 so a score threshold is meaningful.
        scores = self.model.predict(
            [(query, p) for p in passages],
            activation_fn=torch.nn.Sigmoid(),
            show_progress_bar=False,
        )
        return [float(s) for s in scores]


def load_reranker(
    model_name: str,
    cache_dir: str,
    update: bool = False,
    device: str = "auto",
) -> Reranker:
    """Load a cross-encoder, local-first like load_embedder."""
    quiet_transformers()
    dev = resolve_device(device)
    if update:
        return Reranker(CrossEncoder(model_name, cache_folder=cache_dir, device=dev))
    try:
        return Reranker(
            CrossEncoder(model_name, cache_folder=cache_dir, device=dev, local_files_only=True)
        )
    except OSError:
        return Reranker(CrossEncoder(model_name, cache_folder=cache_dir, device=dev))
