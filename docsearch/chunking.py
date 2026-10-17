from __future__ import annotations

import re
from collections.abc import Callable

# Word split by a hyphen at a line end: "disponi-\nbilidade" -> "disponibilidade".
_HYPHEN_BREAK = re.compile(r"(\w)-\n(?=[a-zà-ÿ])")
# Sentence end: punctuation, whitespace, then something that can start a sentence.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[«“]?[A-ZÀ-Ý0-9])")


def clean_page_text(text: str) -> str:
    """Join hyphenated line breaks and collapse whitespace."""
    return " ".join(_HYPHEN_BREAK.sub(r"\1", text).split())


def split_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text) if s]


def _split_long(sentence: str, max_tokens: int, count_tokens: Callable[[str], int]) -> list[str]:
    """Split one over-long sentence into word windows that fit the budget.

    Counts each word once and sums the counts, so the cost is linear in the
    sentence length. The sum can differ slightly from the count of the joined
    text, which is fine for a budget well below the model limit.
    """
    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for word in sentence.split():
        n = count_tokens(word)
        if current and current_tokens + n > max_tokens:
            parts.append(" ".join(current))
            current, current_tokens = [], 0
        current.append(word)
        current_tokens += n
    if current:
        parts.append(" ".join(current))
    return parts


def chunk_text(
    text: str,
    max_tokens: int,
    overlap_sentences: int,
    count_tokens: Callable[[str], int],
) -> list[str]:
    """Pack whole sentences into chunks of at most ``max_tokens`` tokens.

    Each chunk after the first repeats the last ``overlap_sentences`` sentences
    of the previous chunk, so a fact on a sentence boundary is not lost.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences must be >= 0")

    clean = clean_page_text(text)
    if not clean:
        return []

    sentences: list[tuple[str, int]] = []
    for sentence in split_sentences(clean):
        n = count_tokens(sentence)
        if n <= max_tokens:
            sentences.append((sentence, n))
        else:
            sentences.extend(
                (part, count_tokens(part))
                for part in _split_long(sentence, max_tokens, count_tokens)
            )

    chunks: list[str] = []
    current: list[tuple[str, int]] = []
    current_tokens = 0
    fresh = 0  # sentences in `current` not already emitted in a previous chunk
    for sentence, n in sentences:
        if current and current_tokens + n > max_tokens:
            chunks.append(" ".join(s for s, _ in current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_tokens = sum(t for _, t in current)
            # Drop overlap that would not leave room for the new sentence.
            while current and current_tokens + n > max_tokens:
                current_tokens -= current.pop(0)[1]
            fresh = 0
        current.append((sentence, n))
        current_tokens += n
        fresh += 1
    if current and fresh:
        chunks.append(" ".join(s for s, _ in current))
    return chunks
