from .backend_base import OCRBackend
from .backends import DoclingBackend, LightOnOCRBackend
from .schema import ExtractionResult

BACKEND_REGISTRY: dict[str, type[OCRBackend]] = {
    "docling": DoclingBackend,
    "lighton": LightOnOCRBackend,
}


def get_ocr_backend(name: str = "docling") -> OCRBackend:
    """Instantiate an OCR backend by name.

    Args:
        name: Key in BACKEND_REGISTRY (default: "docling").

    Returns:
        An instance of the requested OCRBackend.

    Raises:
        ValueError: If the name is not registered.
    """
    cls = BACKEND_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown OCR backend '{name}'. "
            f"Available: {list(BACKEND_REGISTRY.keys())}"
        )
    return cls()


class DocProcessor:
    def __init__(self, backend: str | OCRBackend = "docling"):
        """Create a document processor with the given OCR backend.

        Args:
            backend: Either a backend name string (looked up in
                     BACKEND_REGISTRY) or an OCRBackend instance.
        """
        if isinstance(backend, str):
            self.backend = get_ocr_backend(backend)
        else:
            self.backend = backend

    def process_document(self, source_pdf_path: str) -> ExtractionResult:
        """
        Process a PDF and return an ExtractionResult.
        If processing fails, returns an empty result instead of raising an exception.
        Corrupt documents shouldn't crash the entire pipeline.
        """
        try:
            return self.backend.extract(source_pdf_path)
        except Exception as e:
            print(f"Failed to process document {source_pdf_path}: {e}")
            return ExtractionResult(
                doc_id="failed_doc",
                source_path=source_pdf_path,
                markdown="",
                source_chunks=[],
                images=[],
                tables=[],
                equations=[],
            )
