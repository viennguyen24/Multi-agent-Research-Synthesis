from abc import ABC, abstractmethod

from .schema import ExtractionResult


class OCRBackend(ABC):
    """Interface for document-to-ExtractionResult converters.

    Every OCR backend must implement the `extract` method, which takes a path
    to a PDF file and returns a fully populated `ExtractionResult`.

    To add a new backend:
    1. Create a new module in `backends/` that subclasses `OCRBackend`.
    2. Register it in `processor.BACKEND_REGISTRY`.
    """

    @abstractmethod
    def extract(self, source_pdf_path: str) -> ExtractionResult:
        """Parse a PDF and return a fully populated ExtractionResult."""
        ...
