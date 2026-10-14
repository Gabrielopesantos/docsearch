# docsearch

**Hybrid keyword + semantic search over local documents**  
*Currently supports PDF files only*

Index your documents locally, search them semantically with natural language queries, all without sending data to external services.

## Requirements

### System dependencies

- **Python 3.11+**
- C++ standard library (`libstdc++`) — for PyTorch/sentence-transformers
- System libraries for PyMuPDF — usually available via package manager

### Python dependencies (auto-installed)

- `libsql` — local vector database
- `pymupdf` — PDF text extraction  
- `sentence-transformers` — local embedding and reranker models (~470MB + ~470MB on first run)
- `torch` + dependencies — pulled in by sentence-transformers (~900MB)

## Installation

### Via Nix (recommended for reproducible dev environment)

```bash
# Clone the repo
git clone <repo-url>
cd docsearch

# Enter dev shell (provides Python, uv, system deps)
nix develop

# Install Python dependencies
uv sync

# Run the CLI
uv run docsearch --help
```

### Without Nix

```bash
# Ensure Python 3.11+ and system deps are installed
python --version

# Install project
pip install -e .

# Or use uv
uv sync
uv run docsearch --help
```

## Quick start

```bash
# Index all PDFs in a directory (or a single PDF file)
docsearch index ~/Documents
docsearch index ~/Documents/paper.pdf

# Search indexed content
docsearch search "what does the SLA policy say?"

# View index stats and document listing
docsearch status

# Remove index entries for files no longer on disk
docsearch sync
```

## How search works

```
index:  PDF ─► text per page ─► strip repeated headers/footers
            ─► sentence chunks (≤250 tokens, never cross a page)
            ─► e5 embeddings + FTS5 keyword index

search: query ─┬─► FTS5 bm25 top 50 ───┐
               │                        ├─► RRF merge ─► cross-encoder rerank ─► top N
               └─► vector top 50 ──────┘
```

- **Keyword search** finds exact terms, IDs and numbers ("A.8.2", "SEC-4471").
  Accents are folded, so "servico" matches "serviço".
- **Vector search** finds passages with the same meaning but different words.
- **The reranker** reads query and passage together and decides the final order.

## Configuration

On first run, a default config file is created at:

```
~/.config/docsearch/config.toml
```

See [`config.toml.example`](./config.toml.example) for the full template with comments.

**Config precedence:** CLI flags > config file > built-in defaults

### Key settings

| Setting | Default | Description |
|---------|---------|-------------|
| `embedding.model_name` | `intfloat/multilingual-e5-small` | HuggingFace model identifier |
| `embedding.batch_size` | `64` | Chunks embedded per batch |
| `embedding.device` | `auto` | `auto`, `cpu` or `cuda` (speed only) |
| `chunking.max_tokens` | `250` | Maximum tokens per chunk |
| `chunking.overlap_sentences` | `1` | Sentences repeated between chunks |
| `search.top_k` | `5` | Default search result count |
| `search.score_threshold` | unset | Minimum score in 0..1 |
| `rerank.enabled` | `true` | Rerank with a cross-encoder |
| `rerank.model_name` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Cross-encoder model |
| `storage.db_path` | `~/.docsearch/index.db` | libSQL database location |

## Usage

### Index PDFs

```bash
# Index a directory (skips already-indexed files by hash)
docsearch index ./pdfs

# Index a single PDF file
docsearch index ./pdfs/paper.pdf

# Force re-index (replaces existing entries for changed files)
docsearch index --reindex ./pdfs
docsearch index --reindex ./pdfs/paper.pdf

# Force model re-download from HuggingFace Hub
docsearch --update-model index ./pdfs
```

### Search

```bash
# Basic search (returns top 5 by default)
docsearch search "neural networks"

# Customize result count
docsearch search "machine learning" -n 10

# Filter by minimum score (0.0 - 1.0; reranker score when rerank is on)
docsearch search "embeddings" --score-threshold 0.3

# Skip the reranker (faster, less precise)
docsearch search --no-rerank "embeddings"

# Limit search to a specific document (by path or hash prefix)
docsearch search --doc /path/to/paper.pdf "query"
docsearch search --doc a3f9c2b1d408 "query"

# Multiple documents (repeat --doc)
docsearch search --doc <hash1> --doc <hash2> "query"

# JSON output for piping to other tools
docsearch -o json search "query" | jq '.results[0]'
```

### Measure search quality

Write an eval file with queries and the page where the answer lives
(see [`eval.example.toml`](./eval.example.toml)), then:

```bash
docsearch eval eval.toml
docsearch eval --no-rerank eval.toml   # compare without the reranker
docsearch -o json eval eval.toml
```

It prints recall@1, recall@5 and MRR (mean of 1/rank of the right page), and
lists the queries that missed. Run it after any change to model, chunking or
settings to see if results got better or worse.

### Status and maintenance

```bash
# Show index stats + per-document breakdown (includes short hash for each document)
docsearch status

# JSON output (includes file_hash field per document)
docsearch -o json status
docsearch -o json status | jq '.documents[].file_hash'

# Remove index entries for deleted PDFs
docsearch sync

# Manually remove a specific file
docsearch remove /path/to/file.pdf
docsearch remove <sha256-hash>
```

## Output formats

All commands support `--output` / `-o`:

- `pretty` (default) — colorized ANSI output with formatting
- `json` — machine-parseable JSON for scripting

```bash
docsearch -o json search "query" | jq .
docsearch -o pretty status
```

## Models

The default **embedding model** is **`intfloat/multilingual-e5-small`**:
- Multilingual (Portuguese, English and ~100 other languages)
- 384 dimensions, ~470MB download, fast on CPU

The default **reranker** is **`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`**
(multilingual, ~470MB). It adds a fraction of a second per query on CPU.

With a GPU, stronger models are practical:

```toml
[embedding]
model_name = "BAAI/bge-m3"
device = "cuda"

[rerank]
model_name = "BAAI/bge-reranker-v2-m3"
```

The index remembers which embedding model built it. If the configured model
differs, `index`, `search` and `eval` refuse to run instead of mixing vectors.
To change models, use a new `--db` or delete the index file and index again.
The device does not affect results, only speed.

### Scanned PDFs

PDFs without a text layer show up as "No text" when indexing. Run OCR first,
for example with [`ocrmypdf`](https://ocrmypdf.readthedocs.io/):

```bash
nix run nixpkgs#ocrmypdf -- scanned.pdf scanned-ocr.pdf
```

### Model download behavior

- **First run:** Models download from HuggingFace Hub (~1GB + ~900MB PyTorch deps)
- **Subsequent runs:** Fully offline, no network requests
- **Force update:** Use `--update-model` flag to re-download

### HuggingFace authentication (optional)

Set `HF_TOKEN` to avoid rate limits during initial model download:

```bash
export HF_TOKEN="hf_..."
docsearch index ./pdfs
```

This is **optional** — the model is public and downloads without auth.

## Database location

By default, the index is stored at:

```
~/.docsearch/index.db
```

Override via config file or `--db` flag:

```bash
docsearch --db ./custom.db status
```

## Development

```bash
# Enter dev shell (if using Nix)
nix develop

# Install in editable mode
uv sync

# Run tests (if any)
pytest

# Lint / format
ruff check .
ruff format .
```

## License

MIT License
