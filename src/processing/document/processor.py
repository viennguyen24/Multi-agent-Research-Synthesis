import logging
from typing import Any

from .pipeline import extract_multimodal_pdf_artifacts
from .schema import ExtractionManifest, ExtractionResult

logger = logging.getLogger(__name__)

class DocProcessor:
    def process_document(self, source_pdf_path: str) -> ExtractionResult:
        """
        Process a PDF and return an ExtractionResult.
        If processing fails, returns an empty result instead of raising an exception. Corrupt document shouldn't crash the entire pipeline.
        """
        try:
            return extract_multimodal_pdf_artifacts(source_pdf_path)
        except Exception as e:
            print(f"Failed to process document {source_pdf_path}: {e}")
            return ExtractionResult(
                source_chunks=[],
                manifest_json=ExtractionManifest(
                    doc_id="failed_doc",
                    source_pdf_path=source_pdf_path,
                    markdown_path="",
                    images=[],
                    tables=[],
                    equations=[],
                    references=[]
                ),
                image_count=0,
                table_count=0,
                equation_count=0,
                chunk_count=0
            )
