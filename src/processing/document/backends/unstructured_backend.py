import base64
import os
from pathlib import Path

from unstructured_client import UnstructuredClient
from unstructured_client.models import shared, operations
from unstructured_client.models.errors import SDKError

from ..backend_base import OCRBackend
from ..schema import (
    ExtractedChunk,
    ExtractedEquation,
    ExtractedImage,
    ExtractedTable,
    ExtractionResult,
)


class UnstructuredBackend(OCRBackend):
    """
    OCR Backend powered by Unstructured.io (Serverless API).
    Handles PDF conversion, contextual chunking, and artifact extraction.
    """

    def __init__(self) -> None:
        api_key = os.getenv("UNSTRUCTURED_API_KEY", "")
        server_url = os.getenv("UNSTRUCTURED_API_URL")
        
        if not api_key:
            print("[UnstructuredBackend] WARNING: UNSTRUCTURED_API_KEY is not set.")

        kwargs = {"api_key_auth": api_key}
        if server_url:
            # Only override if explicitly provided
            kwargs["server_url"] = server_url

        self.client = UnstructuredClient(**kwargs)

    def _build_markdown(self, elements: list[dict]) -> str:
        """Constructs a full markdown representation from extracted elements."""
        return "\n\n".join([el.get("text", "") for el in elements if el.get("text")])

    def _extract_multimodal_artifacts(
        self, elements: list[dict], doc_id: str
    ) -> tuple[list[ExtractedChunk], list[ExtractedImage], list[ExtractedTable], list[ExtractedEquation]]:
        """Parses Unstructured elements into strongly-typed multimodal artifacts."""
        chunks: list[ExtractedChunk] = []
        images: list[ExtractedImage] = []
        tables: list[ExtractedTable] = []
        equations: list[ExtractedEquation] = []

        chunk_idx = 1
        img_idx = 1
        tbl_idx = 1
        eq_idx = 1

        for el in elements:
            el_type = el.get("type", "")
            el_text = el.get("text", "")
            metadata = el.get("metadata", {})
            
            page_numbers = []
            page_no = metadata.get("page_number")
            if page_no:
                page_numbers = [page_no]

            image_base64 = metadata.get("image_base64")
            image_mime = metadata.get("image_mime_type", "image/png")
            
            if el_type == "Image":
                images.append(
                    ExtractedImage(
                        id=f"{doc_id}_img_{img_idx:03d}",
                        mime_type=image_mime,
                        base64_data=image_base64 or "",
                        page=page_no,
                        caption=el_text,
                        contextualized_text=el_text,
                    )
                )
                img_idx += 1

            elif el_type in ("Table", "TableChunk"):
                text_as_html = metadata.get("text_as_html", "")
                tables.append(
                    ExtractedTable(
                        id=f"{doc_id}_tbl_{tbl_idx:03d}",
                        content=text_as_html or el_text,
                        page=page_no,
                        title=f"Table {tbl_idx}",
                        contextualized_text=el_text,
                    )
                )
                tbl_idx += 1

            elif el_type == "Formula":
                equations.append(
                    ExtractedEquation(
                        id=f"{doc_id}_eq_{eq_idx:03d}",
                        latex_or_text=el_text,
                        display_mode="block",
                        page=page_no,
                        contextualized_text=el_text,
                    )
                )
                eq_idx += 1

            else:
                # Handle text chunks (CompositeElement generally preferred for RAG context)
                if el_text:
                    chunks.append(
                        ExtractedChunk(
                            id=f"{doc_id}_chunk_{chunk_idx:04d}",
                            text=el_text,
                            contextualized_text=el_text,
                            meta_data={
                                "headings": [],
                                "captions": [],
                                "page_numbers": page_numbers,
                                "chunk_index": chunk_idx,
                            },
                        )
                    )
                    chunk_idx += 1

        return chunks, images, tables, equations

    def _call_unstructured_api(self, source_path: str) -> list[dict]:
        """Sends the document to the Unstructured Serverless API and returns the parsed elements."""
        source = Path(source_path)
        try:
            with open(source_path, "rb") as f:
                file_content = f.read()

            files = shared.Files(
                content=file_content,
                file_name=source.name,
            )

            params = shared.PartitionParameters(
                files=files,
                strategy="hi_res",
                extract_image_block_types=["Image", "Table", "Formula"],
                extract_image_block_to_payload=True,
                image_description=True,
                table_to_html=True,
                chunking_strategy="by_title",
                contextual_chunking=True,
                max_characters=1500,
            )

            req = operations.PartitionRequest(partition_parameters=params)

            print(f"[UnstructuredBackend] Sending {source.name} to Unstructured API...")
            res = self.client.general.partition(request=req)
            return res.elements or []

        except SDKError as e:
            raise RuntimeError(f"Unstructured API Error: {e.status_code} - {e.body}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to process with UnstructuredBackend: {e}") from e

    def extract(self, source_path: str) -> ExtractionResult:
        """
        Orchestrates the full extraction pipeline using the Unstructured API.
        """
        source = Path(source_path)
        doc_id = source.stem

        elements = self._call_unstructured_api(source_path)
        chunks, images, tables, equations = self._extract_multimodal_artifacts(elements, doc_id)
        markdown = self._build_markdown(elements)

        return ExtractionResult(
            doc_id=doc_id,
            source_path=str(source),
            markdown=markdown,
            source_chunks=chunks,
            images=images,
            tables=tables,
            equations=equations,
            page_count=max([el.get("metadata", {}).get("page_number", 0) for el in elements] + [0]),
            schema="unstructured_api_v1",
        )
