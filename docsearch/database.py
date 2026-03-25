from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import libsql


@dataclass(frozen=True)
class DocumentRecord:
    id: int
    file_path: str
    file_hash: str
    file_name: str
    page_count: int


@dataclass(frozen=True)
class SearchResult:
    file_path: str
    file_name: str
    page_number: int
    chunk_index: int
    content: str
    cosine_distance: float


class Database:
    def __init__(self, db_path: str, vector_dim: int = 384) -> None:
        self.db_path = str(Path(db_path).expanduser())
        self.vector_dim = vector_dim
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = libsql.connect(self.db_path)

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
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
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                page_number INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding F32_BLOB({self.vector_dim}) NOT NULL
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
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id
            ON chunks (document_id)
            """
        )
        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_vector
            ON chunks (libsql_vector_idx(embedding))
            """
        )
        self.conn.commit()

    def get_document_by_hash(self, file_hash: str) -> DocumentRecord | None:
        row = self.conn.execute(
            """
            SELECT id, file_path, file_hash, file_name, page_count
            FROM documents
            WHERE file_hash = ?
            """,
            (file_hash,),
        ).fetchone()
        if row is None:
            return None
        return DocumentRecord(
            id=int(row[0]),
            file_path=str(row[1]),
            file_hash=str(row[2]),
            file_name=str(row[3]),
            page_count=int(row[4]),
        )

    def get_document_by_path(self, file_path: str) -> DocumentRecord | None:
        row = self.conn.execute(
            """
            SELECT id, file_path, file_hash, file_name, page_count
            FROM documents
            WHERE file_path = ?
            """,
            (file_path,),
        ).fetchone()
        if row is None:
            return None
        return DocumentRecord(
            id=int(row[0]),
            file_path=str(row[1]),
            file_hash=str(row[2]),
            file_name=str(row[3]),
            page_count=int(row[4]),
        )

    def delete_document(self, document_id: int) -> None:
        self.conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        self.conn.commit()

    def delete_document_by_path_or_hash(self, key: str) -> int:
        cur = self.conn.execute(
            "DELETE FROM documents WHERE file_path = ? OR file_hash = ?", (key, key)
        )
        self.conn.commit()
        return int(cur.rowcount or 0)

    def insert_document(
        self,
        file_path: str,
        file_hash: str,
        file_name: str,
        page_count: int,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO documents (file_path, file_hash, file_name, page_count)
            VALUES (?, ?, ?, ?)
            """,
            (file_path, file_hash, file_name, page_count),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def insert_chunks(
        self,
        document_id: int,
        rows: Iterable[tuple[int, int, str, list[float]]],
    ) -> int:
        inserted = 0
        for chunk_index, page_number, content, embedding in rows:
            self.conn.execute(
                """
                INSERT INTO chunks (document_id, chunk_index, page_number, content, embedding)
                VALUES (?, ?, ?, ?, vector32(?))
                """,
                (
                    document_id,
                    chunk_index,
                    page_number,
                    content,
                    json.dumps(embedding),
                ),
            )
            inserted += 1
        self.conn.commit()
        return inserted

    def resolve_document_ids(self, keys: list[str]) -> list[int]:
        """Resolve path-or-hash strings to document IDs. Raises ValueError for unknown keys."""
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

    def search_chunks(
        self,
        query_embedding: list[float],
        top_k: int,
        document_ids: list[int] | None = None,
    ) -> list[SearchResult]:
        query_json = json.dumps(query_embedding)
        if document_ids is None:
            rows = self.conn.execute(
                """
                SELECT
                    d.file_path,
                    d.file_name,
                    c.page_number,
                    c.chunk_index,
                    c.content,
                    vector_distance_cos(c.embedding, vector32(?)) AS cosine_distance
                FROM vector_top_k('idx_chunks_vector', vector32(?), ?) AS vt
                JOIN chunks c ON c.rowid = vt.id
                JOIN documents d ON d.id = c.document_id
                ORDER BY cosine_distance ASC
                """,
                (query_json, query_json, top_k),
            ).fetchall()
        else:
            vector_top_k = max(top_k * 10, 200)
            placeholders = ",".join("?" for _ in document_ids)
            rows = self.conn.execute(
                f"""
                SELECT
                    d.file_path,
                    d.file_name,
                    c.page_number,
                    c.chunk_index,
                    c.content,
                    vector_distance_cos(c.embedding, vector32(?)) AS cosine_distance
                FROM vector_top_k('idx_chunks_vector', vector32(?), ?) AS vt
                JOIN chunks c ON c.rowid = vt.id
                JOIN documents d ON d.id = c.document_id
                WHERE c.document_id IN ({placeholders})
                ORDER BY cosine_distance ASC
                LIMIT ?
                """,
                (query_json, query_json, vector_top_k, *document_ids, top_k),
            ).fetchall()
        return [
            SearchResult(
                file_path=str(row[0]),
                file_name=str(row[1]),
                page_number=int(row[2]),
                chunk_index=int(row[3]),
                content=str(row[4]),
                cosine_distance=float(row[5]),
            )
            for row in rows
        ]

    def list_all_documents(self) -> list[DocumentRecord]:
        rows = self.conn.execute(
            "SELECT id, file_path, file_hash, file_name, page_count FROM documents ORDER BY id"
        ).fetchall()
        return [
            DocumentRecord(
                id=int(r[0]),
                file_path=str(r[1]),
                file_hash=str(r[2]),
                file_name=str(r[3]),
                page_count=int(r[4]),
            )
            for r in rows
        ]

    def list_documents_with_chunk_counts(self) -> list[tuple[str, str, str, int, int, str]]:
        """Return (file_name, file_path, file_hash, page_count, chunk_count, indexed_at) per doc."""
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
        return [(str(r[0]), str(r[1]), str(r[2]), int(r[3]), int(r[4]), str(r[5])) for r in rows]

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
