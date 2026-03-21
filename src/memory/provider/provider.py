from __future__ import annotations

from abc import ABC, abstractmethod

from src.processing.document.schema import ExtractionResult


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
    def save_document(self, result: ExtractionResult) -> None:
        """Persist an ExtractionResult to the database."""
        pass

    @abstractmethod
    def load_document(self, doc_id: str) -> ExtractionResult | None:
        """Reconstruct an ExtractionResult from the database. Returns None if not found."""
        pass
