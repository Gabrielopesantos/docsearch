# docsearch

**Semantic search over local documents using embeddings**  
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
- `sentence-transformers` — local embedding model (downloads ~90MB model on first run)
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
# Index all PDFs in a directory
docsearch index ~/Documents

# Search indexed content
docsearch search "what does the SLA policy say?"

# View index stats and document listing
docsearch status

# Remove index entries for files no longer on disk
docsearch sync
```

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
| `embedding.model_name` | `all-MiniLM-L6-v2` | HuggingFace model identifier |
| `embedding.batch_size` | `64` | Chunks embedded per batch |
| `chunking.chunk_size` | `500` | Characters per chunk |
| `chunking.chunk_overlap` | `100` | Overlap between chunks |
| `search.top_k` | `5` | Default search result count |
| `storage.db_path` | `~/.docsearch/index.db` | libSQL database location |

## Usage

### Index PDFs

```bash
# Index a directory (skips already-indexed files by hash)
docsearch index ./pdfs

# Force re-index (replaces existing entries for changed files)
docsearch index --reindex ./pdfs

# Force model re-download from HuggingFace Hub
docsearch --update-model index ./pdfs
```

### Search

```bash
# Basic search (returns top 5 by default)
docsearch search "neural networks"

# Customize result count
docsearch search "machine learning" -n 10

# Filter by minimum similarity score (0.0 - 1.0)
docsearch search "embeddings" --score-threshold 0.7

# JSON output for piping to other tools
docsearch -o json search "query" | jq '.results[0]'
```

### Status and maintenance

```bash
# Show index stats + per-document breakdown
docsearch status

# JSON output
docsearch -o json status

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

## Embedding model

The default model is **`all-MiniLM-L6-v2`** from sentence-transformers:
- 384 dimensions
- ~90MB download
- Fast inference on CPU
- Good quality for general-purpose semantic search

### Model download behavior

- **First run:** Model downloads from HuggingFace Hub (~90MB + ~900MB PyTorch deps)
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
