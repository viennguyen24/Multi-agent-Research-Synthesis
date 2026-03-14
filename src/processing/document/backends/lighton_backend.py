from ..backend_base import OCRBackend
from ..schema import ExtractionResult


class LightOnOCRBackend(OCRBackend):
    """Stub backend for LightOnOCR-2-1B.

    This backend is not yet implemented. When ready, install the
    LightOnOCR-2-1B model and complete the `extract` method to
    produce an `ExtractionResult` matching the shared schema.
    """

    def extract(self, source_pdf_path: str) -> ExtractionResult:
        raise NotImplementedError(
            "LightOnOCR-2-1B backend is not yet implemented. "
            "Install the model and complete this class to enable it."
        )
