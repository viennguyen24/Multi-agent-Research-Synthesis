from __future__ import annotations

import json
import sqlite3
import hashlib
from pathlib import Path
from typing import Any

import sqlite_vec

from src.processing.document.schema import (
    ExtractionResult,
    ExtractedChunk,
    ExtractedImage,
    ExtractedTable,
    ExtractedEquation,
)
from .models import TABLE_NAMES
from ..provider.provider import DatabaseProvider
from .config import DEFAULT_CONFIG, StorageConfig
from .schema import (
    CREATE_DOCUMENTS_TABLE,
    CREATE_EQUATIONS_TABLE,
    CREATE_IMAGES_TABLE,
    CREATE_INDEXES,
    CREATE_TABLES_TABLE,
    CREATE_TEXT_CHUNKS_TABLE,
    CREATE_TEXT_CHUNKS_VEC_TABLE,
)


class SQLiteDatabase(DatabaseProvider):
    """
    SQLite implementation of the DatabaseProvider.
    Handles persistent document storage and vector search using sqlite-vec.
    """

    def __init__(self, config: StorageConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._conn: sqlite3.Connection | None = None
        self.connect()

    def connect(self) -> None:
        """Opens a connection and initializes schema."""
        if self._conn is not None:
            return

        if self.config.auto_create_dirs:
            self.config.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self.config.db_path),
            check_same_thread=self.config.check_same_thread,
            isolation_level=self.config.isolation_level
        )
        self._conn.row_factory = sqlite3.Row

        self._conn.execute(f"PRAGMA journal_mode={self.config.journal_mode}")
        self._conn.execute(f"PRAGMA foreign_keys={'ON' if self.config.foreign_keys else 'OFF'}")

        self._load_vec_extension()
        self.setup()

    def disconnect(self) -> None:
        """Closes the connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> SQLiteDatabase:
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.disconnect()

    def _load_vec_extension(self) -> None:
        """Loads the sqlite-vec extension."""
        if not self._conn:
            raise ValueError("Database not connected.")
        try:
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
        except Exception as e:
            raise RuntimeError(f"Failed to load sqlite-vec: {e}") from e

    def setup(self) -> None:
        """Creates tables and indexes."""
        statements = [
            CREATE_DOCUMENTS_TABLE,
            CREATE_IMAGES_TABLE,
            CREATE_TABLES_TABLE,
            CREATE_EQUATIONS_TABLE,
            CREATE_TEXT_CHUNKS_TABLE,
            CREATE_TEXT_CHUNKS_VEC_TABLE.format(vec_dimensions=self.config.vec_dimensions),
        ]
        statements.extend(CREATE_INDEXES)

        with self._conn:
            for stmt in statements:
                self._conn.execute(stmt)

    def reset(self) -> None:
        """Drops all tables and recreates them."""
        with self._conn:
            for table in TABLE_NAMES:
                self._conn.execute(f"DROP TABLE IF EXISTS {table}")
        self.setup()

    def write_extraction_result(
        self, result: ExtractionResult, embeddings: Any
    ) -> None:
        """
        Unified entry point for relational + vector writes.
        Executes all inserts within a single transaction.
        """
        content_hash = hashlib.sha256(result.source_path.encode()).hexdigest()

        with self._conn:
            # 1. Document
            self._conn.execute(
                """
                INSERT OR IGNORE INTO documents
                (id, source_path, filename, markdown, page_count, content_hash, docling_schema_version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.doc_id,
                    result.source_path,
                    Path(result.source_path).name,
                    result.markdown,
                    result.page_count,
                    content_hash,
                    result.docling_schema_version,
                )
            )

            # 2. Images
            if result.images:
                self._conn.executemany(
                    """
                    INSERT OR IGNORE INTO images
                    (id, document_id, mime_type, base64_data, page_number, caption, contextualized_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            img.id,
                            result.doc_id,
                            img.mime_type,
                            img.base64_data,
                            img.page,
                            img.caption,
                            img.contextualized_text,
                        )
                        for img in result.images
                    ]
                )

            # 3. Tables
            if result.tables:
                self._conn.executemany(
                    """
                    INSERT OR IGNORE INTO tables
                    (id, document_id, html_content, page_number, caption, contextualized_text, col_count, row_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            tbl.id,
                            result.doc_id,
                            tbl.html_content,
                            tbl.page,
                            tbl.title,
                            tbl.contextualized_text,
                            tbl.col_count,
                            tbl.row_count,
                        )
                        for tbl in result.tables
                    ]
                )

            # 4. Equations
            if result.equations:
                self._conn.executemany(
                    """
                    INSERT OR IGNORE INTO equations
                    (id, document_id, text, page_number, contextualized_text, caption)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            eq.id,
                            result.doc_id,
                            eq.latex_or_text,
                            eq.page,
                            eq.contextualized_text,
                            eq.caption,
                        )
                        for eq in result.equations
                    ]
                )

            # 5. Text Chunks
            if result.source_chunks:
                self._conn.executemany(
                    """
                    INSERT OR IGNORE INTO text_chunks
                    (id, document_id, text, contextualized_text, headings_json,
                     page_numbers_json, doc_item_labels_json, chunk_index)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            chunk.id,
                            result.doc_id,
                            chunk.text,
                            chunk.contextualized_text,
                            json.dumps(chunk.headings),
                            json.dumps(chunk.page_numbers),
                            json.dumps(chunk.doc_item_labels),
                            idx,
                        )
                        for idx, chunk in enumerate(result.source_chunks)
                    ]
                )

            # 6. Vectors
            if embeddings:
                self._write_vectors(embeddings)

    def _write_vectors(self, embeddings: Any) -> None:
        import struct

        all_vecs = (
            embeddings.chunk_embeddings +
            embeddings.image_embeddings +
            embeddings.table_embeddings +
            embeddings.equation_embeddings
        )

        if not all_vecs:
            return

        self._conn.executemany(
            "INSERT INTO text_chunks_vec (chunk_id, embedding) VALUES (?, vec_f32(?))",
            [(id_, struct.pack(f"{len(vec)}f", *vec)) for id_, vec in all_vecs]
        )

    def save_document(self, result: ExtractionResult) -> None:
        """Wrapper for write_extraction_result without embeddings."""
        self.write_extraction_result(result, None)

    def load_document(self, doc_id: str) -> ExtractionResult | None:
        """Reconstructs an ExtractionResult from various tables."""
        row = self._conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if not row:
            return None

        # Load chunks
        chunk_rows = self._conn.execute(
            "SELECT * FROM text_chunks WHERE document_id = ? ORDER BY chunk_index", (doc_id,)
        ).fetchall()
        chunks = [
            ExtractedChunk(
                id=r["id"],
                text=r["text"],
                headings=json.loads(r["headings_json"]),
                page_numbers=json.loads(r["page_numbers_json"]),
                doc_item_labels=json.loads(r["doc_item_labels_json"]),
                contextualized_text=r["contextualized_text"],
            )
            for r in chunk_rows
        ]

        images = [
            ExtractedImage(
                id=r["id"],
                mime_type=r["mime_type"],
                base64_data=r["base64_data"],
                page=r["page_number"],
                caption=r["caption"] or "",
                contextualized_text=r["contextualized_text"],
            )
            for r in self._conn.execute("SELECT * FROM images WHERE document_id = ?", (doc_id,)).fetchall()
        ]

        tables = [
            ExtractedTable(
                id=r["id"],
                html_content=r["html_content"],
                page=r["page_number"],
                title=r["caption"] or "",
                contextualized_text=r["contextualized_text"],
                col_count=r["col_count"],
                row_count=r["row_count"],
            )
            for r in self._conn.execute("SELECT * FROM tables WHERE document_id = ?", (doc_id,)).fetchall()
        ]

        equations = [
            ExtractedEquation(
                id=r["id"],
                latex_or_text=r["text"],
                display_mode="block",
                page=r["page_number"],
                contextualized_text=r["contextualized_text"],
                caption=r["caption"] or "",
            )
            for r in self._conn.execute("SELECT * FROM equations WHERE document_id = ?", (doc_id,)).fetchall()
        ]

        return ExtractionResult(
            doc_id=doc_id,
            source_path=row["source_path"],
            markdown=row["markdown"],
            source_chunks=chunks,
            images=images,
            tables=tables,
            equations=equations,
            page_count=row["page_count"],
            docling_schema_version=row["docling_schema_version"],
        )

    def document_exists(self, doc_id: str) -> bool:
        """Returns True if a document with the given ID already exists."""
        row = self._conn.execute(
            "SELECT 1 FROM documents WHERE id = ? LIMIT 1", (doc_id,)
        ).fetchone()
        return row is not None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise ValueError("Database disconnected.")
        return self._conn
