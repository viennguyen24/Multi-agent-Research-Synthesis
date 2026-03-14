import json
import sqlite3
from abc import ABC, abstractmethod
from typing import Any

from src.processing.document.schema import (
    ArtifactReference,
    ExtractedChunk,
    ExtractedEquation,
    ExtractedImage,
    ExtractedTable,
    ExtractionManifest,
    ExtractionResult,
)


class DatabaseProvider(ABC):
    """
    Abstract interface for document extraction persistence.
    Allows easy swapping of backend implementations (e.g. SQLite, PostgreSQL).
    """

    @abstractmethod
    def setup(self) -> None:
        """Create tables and verify schema if they don't exist."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Drop all existing tables/data to provide a fresh start."""
        pass

    @abstractmethod
    def save_result(self, result: ExtractionResult) -> None:
        """Persist an ExtractionResult to the database."""
        pass

    @abstractmethod
    def load_result(self, doc_id: str) -> ExtractionResult | None:
        """Reconstruct an ExtractionResult from the database. Returns None if not found."""
        pass


class SQLiteProvider(DatabaseProvider):
    """
    SQLite implementation of the DatabaseProvider interface.
    Stores all data in text format to allow easy manual querying with GUI tools.
    """

    def __init__(self, db_path: str = "processor.db"):
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        # Check same_thread=False since this is going into a graph pipeline that might use threads
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def setup(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # documents: master record for the extracted PDF
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    source_pdf_path TEXT,
                    markdown_path TEXT
                )
            ''')

            # images: metadata for extracted images including base64 encoded text instead of binary blobs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS images (
                    doc_id TEXT,
                    id TEXT,
                    mime_type TEXT,
                    base64_data TEXT,
                    page INTEGER,
                    caption TEXT,
                    PRIMARY KEY (doc_id, id),
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                )
            ''')

            # tables: metadata for extracted HTML tables 
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tables (
                    doc_id TEXT,
                    id TEXT,
                    html_content TEXT,
                    page INTEGER,
                    title TEXT,
                    PRIMARY KEY (doc_id, id),
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                )
            ''')

            # equations: metadata for extracted equations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS equations (
                    doc_id TEXT,
                    id TEXT,
                    latex_or_text TEXT,
                    display_mode TEXT,
                    page INTEGER,
                    markdown_anchor TEXT,
                    PRIMARY KEY (doc_id, id),
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                )
            ''')

            # chunks: text chunks from the parser
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    doc_id TEXT,
                    chunk_index INTEGER,
                    text TEXT,
                    contextualized_text TEXT,
                    headings_json TEXT,  -- JSON serialized list of strings
                    captions_json TEXT,  -- JSON serialized list of strings
                    PRIMARY KEY (doc_id, chunk_index),
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                )
            ''')

            # references: mapping tokens back to artifacts
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS references_map (
                    doc_id TEXT,
                    token TEXT,
                    item_id TEXT,
                    kind TEXT,
                    PRIMARY KEY (doc_id, token),
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                )
            ''')
            conn.commit()

    def reset(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS references_map")
            cursor.execute("DROP TABLE IF EXISTS chunks")
            cursor.execute("DROP TABLE IF EXISTS equations")
            cursor.execute("DROP TABLE IF EXISTS tables")
            cursor.execute("DROP TABLE IF EXISTS images")
            cursor.execute("DROP TABLE IF EXISTS documents")
            conn.commit()
        self.setup()

    def save_result(self, result: ExtractionResult) -> None:
        manifest = result.manifest_json
        doc_id = manifest.doc_id

        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Save document manifest base
            cursor.execute(
                "INSERT OR REPLACE INTO documents (doc_id, source_pdf_path, markdown_path) VALUES (?, ?, ?)",
                (doc_id, manifest.source_pdf_path, manifest.markdown_path)
            )

            # Save Images
            for img in manifest.images:
                cursor.execute(
                    "INSERT OR REPLACE INTO images (doc_id, id, mime_type, base64_data, page, caption) VALUES (?, ?, ?, ?, ?, ?)",
                    (doc_id, img.id, img.mime_type, img.base64_data, img.page, img.caption)
                )

            # Save Tables
            for tbl in manifest.tables:
                cursor.execute(
                    "INSERT OR REPLACE INTO tables (doc_id, id, html_content, page, title) VALUES (?, ?, ?, ?, ?)",
                    (doc_id, tbl.id, tbl.html_content, tbl.page, tbl.title)
                )

            # Save Equations
            for eq in manifest.equations:
                cursor.execute(
                    "INSERT OR REPLACE INTO equations (doc_id, id, latex_or_text, display_mode, page, markdown_anchor) VALUES (?, ?, ?, ?, ?, ?)",
                    (doc_id, eq.id, eq.latex_or_text, eq.display_mode, eq.page, eq.markdown_anchor)
                )

            # Save References
            for ref in manifest.references:
                cursor.execute(
                    "INSERT OR REPLACE INTO references_map (doc_id, token, item_id, kind) VALUES (?, ?, ?, ?)",
                    (doc_id, ref.token, ref.item_id, ref.kind)
                )

            # Save Chunks
            for idx, chunk in enumerate(result.source_chunks):
                cursor.execute(
                    "INSERT OR REPLACE INTO chunks (doc_id, chunk_index, text, contextualized_text, headings_json, captions_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        doc_id,
                        idx,
                        chunk.text,
                        chunk.contextualized_text,
                        json.dumps(chunk.headings),
                        json.dumps(chunk.captions)
                    )
                )
            
            conn.commit()

    def load_result(self, doc_id: str) -> ExtractionResult | None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Verify document exists
            cursor.execute("SELECT source_pdf_path, markdown_path FROM documents WHERE doc_id = ?", (doc_id,))
            doc_row = cursor.fetchone()
            if not doc_row:
                return None
                
            # Load Images
            cursor.execute("SELECT id, mime_type, base64_data, page, caption FROM images WHERE doc_id = ?", (doc_id,))
            images = [
                ExtractedImage(
                    id=row["id"],
                    mime_type=row["mime_type"],
                    base64_data=row["base64_data"],
                    page=row["page"],
                    caption=row["caption"]
                )
                for row in cursor.fetchall()
            ]

            # Load Tables
            cursor.execute("SELECT id, html_content, page, title FROM tables WHERE doc_id = ?", (doc_id,))
            tables = [
                ExtractedTable(
                    id=row["id"],
                    html_content=row["html_content"],
                    page=row["page"],
                    title=row["title"]
                )
                for row in cursor.fetchall()
            ]

            # Load Equations
            cursor.execute("SELECT id, latex_or_text, display_mode, page, markdown_anchor FROM equations WHERE doc_id = ?", (doc_id,))
            equations = [
                ExtractedEquation(
                    id=row["id"],
                    latex_or_text=row["latex_or_text"],
                    display_mode=row["display_mode"],
                    page=row["page"],
                    markdown_anchor=row["markdown_anchor"]
                )
                for row in cursor.fetchall()
            ]

            # Load References
            cursor.execute("SELECT token, item_id, kind FROM references_map WHERE doc_id = ?", (doc_id,))
            references = [
                ArtifactReference(token=row["token"], item_id=row["item_id"], kind=row["kind"])
                for row in cursor.fetchall()
            ]

            # Reconstruct Manifest
            manifest = ExtractionManifest(
                doc_id=doc_id,
                source_pdf_path=doc_row["source_pdf_path"],
                markdown_path=doc_row["markdown_path"],
                images=images,
                tables=tables,
                equations=equations,
                references=references
            )

            # Load Chunks
            cursor.execute("SELECT text, contextualized_text, headings_json, captions_json FROM chunks WHERE doc_id = ? ORDER BY chunk_index ASC", (doc_id,))
            source_chunks = []
            for row in cursor.fetchall():
                headings = json.loads(row["headings_json"]) if row["headings_json"] else []
                captions = json.loads(row["captions_json"]) if row["captions_json"] else []
                source_chunks.append(
                    ExtractedChunk(
                        text=row["text"],
                        contextualized_text=row["contextualized_text"],
                        headings=headings,
                        captions=captions
                    )
                )

            return ExtractionResult(
                source_chunks=source_chunks,
                manifest_json=manifest,
                image_count=len(images),
                table_count=len(tables),
                equation_count=len(equations),
                chunk_count=len(source_chunks)
            )


def get_database_provider() -> DatabaseProvider:
    """Helper method to construct the active database provider."""
    return SQLiteProvider()
