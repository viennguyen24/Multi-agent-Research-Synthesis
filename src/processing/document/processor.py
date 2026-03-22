from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .backend_base import OCRBackend
from .backends import DoclingBackend, LightOnOCRBackend
from .schema import ExtractionResult

if TYPE_CHECKING:
    from ...memory.sqlite.database import SQLiteDatabase
    from ..context.contextualizer import GeminiContextualizer

BACKEND_REGISTRY: dict[str, type[OCRBackend]] = {
    "docling": DoclingBackend,
    "lighton": LightOnOCRBackend,
}


def get_ocr_backend(name: str = "docling") -> OCRBackend:
    """Instantiate an OCR backend by name."""
    cls = BACKEND_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown OCR backend '{name}'. "
            f"Available: {list(BACKEND_REGISTRY.keys())}"
        )
    return cls()


class DocProcessor:
    def __init__(
        self,
        backend: str | OCRBackend = "docling",
        db: SQLiteDatabase | None = None,
        contextualizer: GeminiContextualizer | None = None,
        embedder: Any | None = None,
    ) -> None:
        """
        Create a document processor with the given OCR backend and optional pipeline stages.
        """
        if isinstance(backend, str):
            self.backend = get_ocr_backend(backend)
        else:
            self.backend = backend
        self._db = db
        self._contextualizer = contextualizer
        self._embedder = embedder

    def process_document(self, source_path: str) -> ExtractionResult | None:
        """
        Full pipeline: extract → contextualize → embed → write.
        Returns ExtractionResult on success, None on failure.
        """
        try:
            doc_id = Path(source_path).stem

            # Skip re-ingestion if document already exists
            if self._db and self._db.document_exists(doc_id):
                cached = self._db.load_document(doc_id)
                if cached:
                    return cached

            # Parse document
            result = self.backend.extract(source_path)

            # Contextualize each chunk in document (mutates result in-place)
            if self._contextualizer:
                result = self._contextualizer.contextualize(result)

            # Create embeddings on document chunks for retrieval
            embeddings = None
            if self._embedder:
                embeddings = self._embedder.embed_extraction_result(result)

            # Persist to database
            if self._db:
                self._db.write_extraction_result(result, embeddings)

            return result

        except Exception as e:
            print(f"[DocProcessor] Failed to ingest {source_path}: {e}")
            return None

    def extract_only(self, source_path: str) -> ExtractionResult | None:
        """Extraction without contextualization or storage. Useful for inspection."""
        try:
            return self.backend.extract(source_path)
        except Exception as e:
            print(f"[DocProcessor] Extraction failed for {source_path}: {e}")
            return None
