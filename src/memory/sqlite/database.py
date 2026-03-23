from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import sqlite_vec

from src.processing.document.schema import (
    ExtractionResult,
    ExtractedChunk,
    ExtractedImage,
    ExtractedTable,
    ExtractedEquation,
)
from ..provider.provider import DatabaseProvider
from .config import DEFAULT_CONFIG, StorageConfig, TABLE_NAMES
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

    def document_exists(self, content_hash: str) -> bool:
        """Returns True if a document with the given content hash already exists."""
        # Using the content_hash to determine if a document exists
        row = self._conn.execute(
            "SELECT 1 FROM documents WHERE content_hash = ? LIMIT 1", (content_hash,)
        ).fetchone()
        return row is not None
        
    def load_document_by_hash(self, content_hash: str) -> ExtractionResult | None:
        """Loads an ExtractionResult from the database using its content hash."""
        row = self._conn.execute(
            "SELECT id FROM documents WHERE content_hash = ? LIMIT 1", (content_hash,)
        ).fetchone()
        if not row:
            return None
        return self.load_document(row["id"])

    def save_document(self, result: ExtractionResult) -> None:
        """Persists an ExtractionResult to the database."""
        doc_id = result.doc_id
        content_hash = result.content_hash

        with self._conn:
            # 1. Save Document
            self._conn.execute(
                """
                INSERT OR REPLACE INTO documents 
                (id, source_path, filename, markdown, page_count, content_hash, docling_schema_version) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    result.source_path,
                    Path(result.source_path).name,
                    result.markdown,
                    result.page_count,
                    content_hash,
                    result.docling_schema_version
                )
            )

            # 2. Save Images
            for img in result.images:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO images 
                    (id, document_id, mime_type, base64_data, page_number, caption, contextualized_text) 
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (img.id, doc_id, img.mime_type, img.base64_data, img.page, img.caption, img.contextualized_text)
                )

            # 3. Save Tables
            for tbl in result.tables:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO tables 
                    (id, document_id, html_content, page_number, caption, contextualized_text, col_count, row_count) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (tbl.id, doc_id, tbl.html_content, tbl.page, tbl.title, tbl.contextualized_text, tbl.col_count, tbl.row_count)
                )

            # 4. Save Equations
            for eq in result.equations:
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO equations 
                    (id, document_id, text, contextualized_text, page_number, caption) 
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (eq.id, doc_id, eq.latex_or_text, eq.contextualized_text, eq.page, eq.caption)
                )

            # 5. Save Text Chunks
            for idx, chunk in enumerate(result.source_chunks):        
                self._conn.execute(
                    """
                    INSERT OR REPLACE INTO text_chunks 
                    (id, document_id, text, headings_json, captions_json, page_numbers_json, chunk_index, contextualized_text) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.id,
                        doc_id,
                        chunk.text,
                        json.dumps(chunk.headings),
                        json.dumps(chunk.captions),
                        json.dumps(chunk.page_numbers),
                        idx,
                        chunk.contextualized_text
                    )
                )

    def load_document(self, doc_id: str) -> ExtractionResult | None:
        """Loads an ExtractionResult from the database."""
        # 1. Load Document
        doc_row = self._conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        if not doc_row:
            return None

        # 2. Load Images
        img_rows = self._conn.execute("SELECT * FROM images WHERE document_id = ?", (doc_id,)).fetchall()
        images = [
            ExtractedImage(
                id=row["id"],
                mime_type=row["mime_type"],
                base64_data=row["base64_data"],
                page=row["page_number"],
                caption=row["caption"] or "",
                contextualized_text=row["contextualized_text"]
            ) for row in img_rows
        ]

        # 3. Load Tables
        tbl_rows = self._conn.execute("SELECT * FROM tables WHERE document_id = ?", (doc_id,)).fetchall()
        tables = [
            ExtractedTable(
                id=row["id"],
                html_content=row["html_content"],
                page=row["page_number"],
                title=row["caption"] or "",
                contextualized_text=row["contextualized_text"],
                col_count=row["col_count"],
                row_count=row["row_count"]
            ) for row in tbl_rows
        ]

        # 4. Load Equations
        eq_rows = self._conn.execute("SELECT * FROM equations WHERE document_id = ?", (doc_id,)).fetchall()
        equations = [
            ExtractedEquation(
                id=row["id"],
                latex_or_text=row["text"],
                display_mode="block", # Defaulting as not stored
                page=row["page_number"],
                caption=row["caption"] or "",
                contextualized_text=row["contextualized_text"]
            ) for row in eq_rows
        ]

        # 5. Load Chunks
        chunk_rows = self._conn.execute(
            "SELECT * FROM text_chunks WHERE document_id = ? ORDER BY chunk_index", 
            (doc_id,)
        ).fetchall()
        source_chunks = [
            ExtractedChunk(
                id=row["id"],
                text=row["text"],
                contextualized_text=row["contextualized_text"],
                headings=json.loads(row["headings_json"]),
                captions=json.loads(row["captions_json"]),
                page_numbers=json.loads(row["page_numbers_json"])
            ) for row in chunk_rows
        ]

        return ExtractionResult(
            doc_id=doc_id,
            source_path=doc_row["source_path"],
            markdown=doc_row["markdown"],
            source_chunks=source_chunks,
            images=images,
            tables=tables,
            equations=equations,
            page_count=doc_row["page_count"],
            docling_schema_version=doc_row["docling_schema_version"]
        )

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise ValueError("Database disconnected.")
        return self._conn
