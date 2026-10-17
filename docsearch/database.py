from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import libsql


class IndexMismatchError(ValueError):
    """The index on disk was built with a different embedding model than the config."""


@dataclass(frozen=True)
class DocumentRecord:
    id: int
    file_path: str
    file_hash: str
    file_name: str
    page_count: int
    indexed_at: str


_DOCUMENT_COLUMNS = "id, file_path, file_hash, file_name, page_count, indexed_at"


def _document_from_row(row: tuple) -> DocumentRecord:
    return DocumentRecord(
        id=int(row[0]),
        file_path=str(row[1]),
        file_hash=str(row[2]),
        file_name=str(row[3]),
        page_count=int(row[4]),
        indexed_at=str(row[5]),
    )


@dataclass(frozen=True)
class ChunkRow:
    id: int
    file_path: str
    file_name: str
    page_number: int
    chunk_index: int
    content: str


@dataclass(frozen=True)
class DocumentSummary:
    """Aggregated document row returned by list_documents_with_chunk_counts."""

    file_name: str
    file_path: str
    file_hash: str
    page_count: int
    chunk_count: int
    indexed_at: str


class Database:
    """libSQL index of documents, chunks, vectors and keywords.

    The first open with a model records its name and vector size in ``meta``
    and creates the chunk tables for that size. Later opens with a model check
    that it matches. Opening without a model (status, sync, remove) needs an
    existing index.
    """

    def __init__(
        self,
        db_path: str,
        model_name: str | None = None,
        vector_dim: int | None = None,
    ) -> None:
        self.db_path = str(Path(db_path).expanduser())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = libsql.connect(self.db_path)
        self.init_schema(model_name, vector_dim)

    def close(self) -> None:
        self.conn.close()

    def init_schema(self, model_name: str | None, vector_dim: int | None) -> None:
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                file_hash TEXT NOT NULL UNIQUE,
                file_name TEXT NOT NULL,
                page_count INTEGER NOT NULL,
                indexed_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_file_hash
            ON documents (file_hash)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        meta = self.get_meta()
        if model_name is not None and vector_dim is not None:
            if not meta:
                meta = {"model_name": model_name, "dim": str(vector_dim)}
                self.conn.executemany(
                    "INSERT INTO meta (key, value) VALUES (?, ?)", list(meta.items())
                )
            elif meta["model_name"] != model_name:
                raise IndexMismatchError(
                    f"index at {self.db_path} was built with {meta['model_name']!r}, "
                    f"config uses {model_name!r}.\nUse --db <new path> for a new index, "
                    "or delete the index file and index again."
                )
        if meta:
            self._create_chunk_tables(int(meta["dim"]))
        self.conn.commit()

    def get_meta(self) -> dict[str, str]:
        rows = self.conn.execute("SELECT key, value FROM meta").fetchall()
        return {str(r[0]): str(r[1]) for r in rows}

    def _create_chunk_tables(self, vector_dim: int) -> None:
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                page_number INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding F32_BLOB({vector_dim}) NOT NULL
            )
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id
            ON chunks (document_id)
            """
        )
        # Keyword index over chunk text. remove_diacritics folds accents, so
        # "servico" matches "serviço". Triggers keep it in sync with chunks,
        # including rows removed by ON DELETE CASCADE.
        self.conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                content='chunks',
                content_rowid='id',
                tokenize='unicode61 remove_diacritics 2'
            )
            """
        )
        self.conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts (rowid, content) VALUES (new.id, new.content);
            END
            """
        )
        self.conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts (chunks_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
            END
            """
        )

    def get_document_by_hash(self, file_hash: str) -> DocumentRecord | None:
        row = self.conn.execute(
            f"SELECT {_DOCUMENT_COLUMNS} FROM documents WHERE file_hash = ?", (file_hash,)
        ).fetchone()
        return None if row is None else _document_from_row(row)

    def get_document_by_path(self, file_path: str) -> DocumentRecord | None:
        row = self.conn.execute(
            f"SELECT {_DOCUMENT_COLUMNS} FROM documents WHERE file_path = ?", (file_path,)
        ).fetchone()
        return None if row is None else _document_from_row(row)

    def now(self) -> str:
        """Current time in the same clock and format as ``documents.indexed_at``."""
        return str(self.conn.execute("SELECT datetime('now')").fetchone()[0])

    def delete_document_by_path_or_hash(self, key: str) -> int:
        cur = self.conn.execute(
            "DELETE FROM documents WHERE file_path = ? OR file_hash = ?", (key, key)
        )
        self.conn.commit()
        return int(cur.rowcount or 0)

    def replace_document(
        self,
        delete_ids: Iterable[int],
        file_path: str,
        file_hash: str,
        file_name: str,
        page_count: int,
        rows: Iterable[tuple[int, int, str, list[float]]],
    ) -> tuple[int, int]:
        """Delete old documents and insert the new one with its chunks in one transaction.

        An interrupt (Ctrl+C) or error rolls back, so the index keeps the old state.
        Returns (document id, chunks inserted).
        """
        try:
            for document_id in delete_ids:
                self.conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            doc_id = self._insert_document(file_path, file_hash, file_name, page_count)
            inserted = self._insert_chunks(doc_id, rows)
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise
        return doc_id, inserted

    def _insert_document(
        self, file_path: str, file_hash: str, file_name: str, page_count: int
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO documents (file_path, file_hash, file_name, page_count)
            VALUES (?, ?, ?, ?)
            """,
            (file_path, file_hash, file_name, page_count),
        )
        return int(cur.lastrowid)

    def _insert_chunks(
        self,
        document_id: int,
        rows: Iterable[tuple[int, int, str, list[float]]],
    ) -> int:
        params = [
            (document_id, chunk_index, page_number, content, json.dumps(embedding))
            for chunk_index, page_number, content, embedding in rows
        ]
        if params:
            self.conn.executemany(
                """
                INSERT INTO chunks (document_id, chunk_index, page_number, content, embedding)
                VALUES (?, ?, ?, ?, vector32(?))
                """,
                params,
            )
        return len(params)

    def resolve_document_ids(self, keys: list[str]) -> list[int]:
        """Resolve path-or-hash strings to document IDs.

        A key is treated as a hash prefix when it is at least 4 characters long and
        contains only hexadecimal digits.  Use absolute paths to avoid ambiguity —
        a relative path such as ``deadbeef.pdf`` would be misclassified as a hash
        prefix.

        Raises ValueError for ambiguous hash prefixes or unknown keys.
        """
        ids: list[int] = []
        unresolved: list[str] = []
        for key in keys:
            if len(key) >= 4 and all(c in "0123456789abcdefABCDEF" for c in key):
                rows = self.conn.execute(
                    "SELECT id FROM documents WHERE file_hash LIKE ? || '%'", (key,)
                ).fetchall()
                if len(rows) > 1:
                    raise ValueError(f"ambiguous hash prefix {key!r} matches {len(rows)} documents")
                if rows:
                    ids.append(int(rows[0][0]))
                else:
                    unresolved.append(key)
            else:
                row = self.conn.execute(
                    "SELECT id FROM documents WHERE file_path = ?", (key,)
                ).fetchone()
                if row is None:
                    unresolved.append(key)
                else:
                    ids.append(int(row[0]))
        if unresolved:
            keys_str = ", ".join(repr(k) for k in unresolved)
            raise ValueError(f"unknown document key(s): {keys_str}")
        return ids

    @staticmethod
    def _doc_filter(document_ids: list[int] | None) -> tuple[str, list[int]]:
        if document_ids is None:
            return "", []
        placeholders = ",".join("?" for _ in document_ids)
        return f"AND c.document_id IN ({placeholders})", list(document_ids)

    def vector_candidates(
        self,
        query_embedding: list[float],
        limit: int,
        document_ids: list[int] | None = None,
    ) -> list[int]:
        """Return chunk ids nearest to the query vector, best first.

        Scans every chunk (or only the filtered documents' chunks), so the result
        is exact. The cost grows linearly with the number of chunks.
        """
        where, params = self._doc_filter(document_ids)
        rows = self.conn.execute(
            f"""
            SELECT c.id
            FROM chunks c
            WHERE 1 = 1 {where}
            ORDER BY vector_distance_cos(c.embedding, vector32(?)) ASC
            LIMIT ?
            """,
            (*params, json.dumps(query_embedding), limit),
        ).fetchall()
        return [int(r[0]) for r in rows]

    def keyword_candidates(
        self,
        match_expr: str,
        limit: int,
        document_ids: list[int] | None = None,
    ) -> list[int]:
        """Return chunk ids that match an FTS5 expression, best bm25 first."""
        where, params = self._doc_filter(document_ids)
        rows = self.conn.execute(
            f"""
            SELECT c.id
            FROM chunks_fts f
            JOIN chunks c ON c.id = f.rowid
            WHERE chunks_fts MATCH ? {where}
            ORDER BY f.rank
            LIMIT ?
            """,
            (match_expr, *params, limit),
        ).fetchall()
        return [int(r[0]) for r in rows]

    def get_chunks(self, chunk_ids: list[int]) -> dict[int, ChunkRow]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self.conn.execute(
            f"""
            SELECT c.id, d.file_path, d.file_name, c.page_number, c.chunk_index, c.content
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id IN ({placeholders})
            """,
            chunk_ids,
        ).fetchall()
        return {
            int(r[0]): ChunkRow(
                id=int(r[0]),
                file_path=str(r[1]),
                file_name=str(r[2]),
                page_number=int(r[3]),
                chunk_index=int(r[4]),
                content=str(r[5]),
            )
            for r in rows
        }

    def list_all_documents(self) -> list[DocumentRecord]:
        rows = self.conn.execute(
            f"SELECT {_DOCUMENT_COLUMNS} FROM documents ORDER BY id"
        ).fetchall()
        return [_document_from_row(r) for r in rows]

    def list_documents_with_chunk_counts(self) -> list[DocumentSummary]:
        """Return one DocumentSummary per indexed document, newest first."""
        rows = self.conn.execute(
            """
            SELECT
                d.file_name,
                d.file_path,
                d.file_hash,
                d.page_count,
                COUNT(c.id) AS chunk_count,
                d.indexed_at
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id
            ORDER BY d.indexed_at DESC
            """
        ).fetchall()
        return [
            DocumentSummary(
                file_name=str(r[0]),
                file_path=str(r[1]),
                file_hash=str(r[2]),
                page_count=int(r[3]),
                chunk_count=int(r[4]),
                indexed_at=str(r[5]),
            )
            for r in rows
        ]

    def delete_documents_batch(self, ids: list[int]) -> None:
        """Delete multiple documents (and their chunks via CASCADE) in a single transaction."""
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        self.conn.execute(f"DELETE FROM documents WHERE id IN ({placeholders})", ids)
        self.conn.commit()

    def stats(self) -> dict[str, int]:
        docs = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return {
            "documents": int(docs[0]) if docs else 0,
            "chunks": int(chunks[0]) if chunks else 0,
        }

    def count_chunks_for_documents(self, document_ids: list[int]) -> int:
        if not document_ids:
            return 0
        placeholders = ",".join("?" for _ in document_ids)
        row = self.conn.execute(
            f"SELECT COUNT(*) FROM chunks WHERE document_id IN ({placeholders})",
            document_ids,
        ).fetchone()
        return int(row[0]) if row else 0
