# docsearch

**Keyword and semantic search over local documents.**

Supports PDF files only. All indexing and search runs locally.

## Requirements

- Python 3.11+
- `libstdc++` (needed by PyTorch)

Python dependencies (installed by `uv sync`):

- `libsql`: local database for chunks, vectors and keywords
- `pymupdf`: PDF text extraction
- `sentence-transformers` and `torch`: local embedding and reranker models

## Installation

With Nix:

```bash
git clone <repo-url>
cd docsearch
nix develop   # Python, uv and system libraries
uv sync
uv run docsearch --help
```

Without Nix, install Python, then:

```bash
uv sync
uv run docsearch --help
```

## Quick start

```bash
docsearch index ~/Documents                         # index all PDFs in a directory
docsearch search "what does the SLA policy say?"
docsearch status                                    # list indexed documents
docsearch sync                                      # remove entries for deleted files
```

## How search works

```bash
index:  PDF ─► text per page ─► strip repeated headers/footers
            ─► sentence chunks (≤250 tokens, never cross a page)
            ─► embeddings + FTS5 keyword index

search: query ─┬─► FTS5 bm25 top 50 ──────┐
               │                          ├─► RRF merge ─► rerank ─► top N
               └─► vector scan top 50 ────┘
```

- **Keyword search** finds exact terms, IDs and numbers ("A.8.2", "SEC-4471").
- **Vector search** finds passages with the same meaning but different words.
  It compares the query with every chunk, so results are exact.
- **The reranker** reads the query and each passage together and sets the final order.

## Configuration

The first run creates `~/.config/docsearch/config.toml`. See
[`config.toml.example`](./config.toml.example) for all settings.
CLI flags override the config file.

| Setting | Default | Description |
|---------|---------|-------------|
| `embedding.model_name` | `intfloat/multilingual-e5-small` | Embedding model |
| `embedding.batch_size` | `64` | Chunks embedded per batch |
| `embedding.device` | `auto` | `auto`, `cpu` or `cuda` (changes speed only) |
| `chunking.max_tokens` | `250` | Maximum tokens per chunk |
| `chunking.overlap_sentences` | `1` | Sentences repeated between chunks |
| `search.top_k` | `5` | Number of results |
| `search.score_threshold` | unset | Minimum score in 0..1 |
| `rerank.enabled` | `true` | Rerank results with a cross-encoder |
| `rerank.model_name` | `BAAI/bge-reranker-v2-m3` | Reranker model |
| `storage.db_path` | `~/.docsearch/index.db` | Index file (or use `--db`) |

## Usage

### Index

```bash
docsearch index ./pdfs                # skips files already in the index
docsearch index ./pdfs/paper.pdf
docsearch index --reindex ./pdfs      # index all files again
docsearch --update-model index ./pdfs # download the models again
```

PDFs without a text layer show "no text". Run OCR on them first, for example:

```bash
nix run nixpkgs#ocrmypdf -- scanned.pdf scanned-ocr.pdf
```

### Search

```bash
docsearch search "machine learning" -n 10           # 10 results
docsearch search "embeddings" --score-threshold 0.3 # minimum score
docsearch search --no-rerank "embeddings"           # faster, less precise
docsearch search --doc /path/to/paper.pdf "query"   # one document (path or hash prefix)
docsearch search --doc <hash1> --doc <hash2> "query"
```

### Measure search quality

Write an eval file with queries and the page that has the answer (see
[`eval.example.toml`](./eval.example.toml)), then:

```bash
docsearch eval eval.toml
docsearch eval --no-rerank eval.toml
```

It shows recall@1, recall@5 and MRR (mean of 1/rank of the right page), and
lists the queries that missed. Run it after you change a model, chunking or
other settings.

### Maintenance

```bash
docsearch status                   # documents, chunk counts and short hashes
docsearch sync                     # remove entries for deleted files
docsearch remove /path/to/file.pdf # remove one file (path or hash)
```

### JSON output

All commands accept `-o json`:

```bash
docsearch -o json search "query" | jq '.results[0]'
```

## Models

Embedding model: **`intfloat/multilingual-e5-small`**. It is multilingual
(Portuguese, English and about 100 other languages), has 384 dimensions and is
fast on CPU.

Reranker: **`BAAI/bge-reranker-v2-m3`**. It is multilingual and adds about
0.5s per query on a GPU. It is slow on CPU. On CPU, use the smaller reranker,
which is faster but less precise:

```toml
[rerank]
model_name = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
```

In a test on 46 English technical books (26 queries), the reranker made the
difference. Recall@1 was 0.58 with the mmarco reranker and 0.77 with
bge-reranker-v2-m3. Bigger embedding models (`BAAI/bge-m3`,
`intfloat/multilingual-e5-large`) did not score better. Run `docsearch eval`
on your own documents before you change models.

The index records which embedding model built it. If the configured model is
different, `index`, `search` and `eval` stop with an error. To change the
embedding model, use a new `--db` or delete the index file and index again.

The models download from Hugging Face on the first run (about 2.7 GB). After
that, docsearch works offline. Set `HF_TOKEN` to avoid download rate limits.

## Development

```bash
nix develop
uv sync
ruff check .
ruff format .
```

## License

MIT
