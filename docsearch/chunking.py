from __future__ import annotations


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    clean = " ".join(text.split())
    if not clean:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    start = 0
    text_len = len(clean)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(clean[start:end])
        if end == text_len:
            break
        start += step

    return chunks
