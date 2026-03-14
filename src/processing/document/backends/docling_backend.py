import base64
import os
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, TypeVar

import psutil
from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument, ImageRefMode

from .._common import _slugify, _verify_references_in_markdown, build_artifact_references
from ..backend_base import OCRBackend
from ..schema import (
    ArtifactReference,
    ExtractedChunk,
    ExtractedEquation,
    ExtractedImage,
    ExtractedTable,
    ExtractionManifest,
    ExtractionResult,
    T,
)

# ---------------------------------------------------------------------------
# Regex constants
# ---------------------------------------------------------------------------

_EQUATION_TOKEN_PATTERN = re.compile(r"\[\[eq:eq_\d{3}\]\]")
_INLINE_EQUATION_PATTERN = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")
_BLOCK_EQUATION_PATTERN = re.compile(r"\$\$(.+?)\$\$", flags=re.DOTALL)
_SECTION_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)\s")


# ---------------------------------------------------------------------------
# Environment / setup helpers
# ---------------------------------------------------------------------------


def _disable_hf_symlink_usage_on_windows() -> None:
    """Force huggingface_hub to copy files instead of symlinking on Windows."""
    if os.name != "nt":
        return

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        import huggingface_hub.file_download as hf_file_download
    except ImportError:
        # Consider logging a warning here if huggingface_hub is an expected dependency.
        return

    hf_file_download._are_symlinks_supported_in_dir.clear()

    def _always_false(_cache_dir: str | Path | None = None) -> bool:
        return False

    hf_file_download.are_symlinks_supported = _always_false


def _build_pdf_pipeline_options() -> PdfPipelineOptions:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_picture_images = True
    pipeline_options.do_formula_enrichment = True
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=psutil.cpu_count(logical=False), device="cpu"
    )
    return pipeline_options


# ---------------------------------------------------------------------------
# Document-level helpers (operate on DoclingDocument)
# ---------------------------------------------------------------------------


def _infer_heading_depth_from_numbering(document: DoclingDocument) -> None:
    """Infer proper heading depth from section numbering."""
    headers = [
        item for item in document.texts
        if "section_header" in str(getattr(item, "label", ""))
    ]

    has_numbered = False
    for item in headers:
        text = (getattr(item, "text", "") or "").strip()
        m = _SECTION_NUMBER_RE.match(text)
        if m:
            item.level = m.group(1).count(".") + 1
            has_numbered = True

    if not has_numbered:
        return

    last_numbered_level = 0
    for item in headers:
        text = (getattr(item, "text", "") or "").strip()
        if _SECTION_NUMBER_RE.match(text):
            last_numbered_level = item.level
        elif last_numbered_level > 1:
            item.level = last_numbered_level + 1


def _extract_page_no(item: Any) -> int | None:
    provs = getattr(item, "prov", None)
    if not provs:
        return None
    first = provs[0]
    page_no = getattr(first, "page_no", None)
    return int(page_no) if isinstance(page_no, int) else None


def _extract_caption(item: Any) -> str:
    captions = getattr(item, "captions", None)
    if not captions:
        return ""
    pieces: list[str] = []
    for cap in captions:
        text = getattr(cap, "text", "")
        if text:
            pieces.append(str(text).strip())
    return " ".join(pieces).strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _extract_chunks(document: DoclingDocument) -> list[ExtractedChunk]:
    chunker = HybridChunker()
    source_chunks: list[ExtractedChunk] = []
    for chunk in chunker.chunk(dl_doc=document):
        source_chunks.append(
            ExtractedChunk(
                text=chunk.text,
                contextualized_text=chunker.contextualize(chunk),
                headings=list(chunk.meta.headings) if chunk.meta.headings else [],
                captions=list(chunk.meta.captions) if chunk.meta.captions else [],
            )
        )
    return source_chunks


# ---------------------------------------------------------------------------
# Artifact extraction
# ---------------------------------------------------------------------------


def _extract_artifact(
    item: Any,
    item_id: str,
    kind: Literal["image", "table"],
    document: DoclingDocument,
) -> ExtractedImage | ExtractedTable | None:
    if kind == "image":
        image = item.get_image(document)
        if image is None:
            return None
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return ExtractedImage(
            id=item_id,
            mime_type="image/png",
            base64_data=img_str,
            page=_extract_page_no(item),
            caption=_extract_caption(item),
        )
    elif kind == "table":
        html = item.export_to_html(doc=document)
        idx = int(item_id.split("_")[-1])
        return ExtractedTable(
            id=item_id,
            html_content=html,
            page=_extract_page_no(item),
            title=f"Table {idx}",
        )
    return None


def _extract_artifacts(
    document: DoclingDocument,
    attr_name: str,
    prefix: str,
    kind: Literal["image", "table"],
    expected_type: type[T],
) -> list[T]:
    extracted_items: list[T] = []
    items = getattr(document, attr_name, [])
    for idx, item in enumerate(items, start=1):
        item_id = f"{prefix}_{idx:03d}"
        res = _extract_artifact(item, item_id, kind, document)
        if res and isinstance(res, expected_type):
            extracted_items.append(res)
    return extracted_items


def _build_image_metadata(document: DoclingDocument) -> list[ExtractedImage]:
    return _extract_artifacts(
        document=document,
        attr_name="pictures",
        prefix="img",
        kind="image",
        expected_type=ExtractedImage,
    )


# ---------------------------------------------------------------------------
# Equation annotation
# ---------------------------------------------------------------------------


def _annotate_equations(markdown_text: str) -> tuple[str, list[ExtractedEquation]]:
    equations: list[ExtractedEquation] = []
    counter = 1

    def block_repl(match: re.Match[str]) -> str:
        nonlocal counter
        expression = match.group(1).strip()
        eq_id = f"eq_{counter:03d}"
        token = f"[[eq:{eq_id}]]"
        equations.append(
            ExtractedEquation(
                id=eq_id,
                latex_or_text=expression,
                display_mode="block",
                page=None,
                markdown_anchor=token,
            )
        )
        counter += 1
        return f"$${expression}$$\n<!-- {token} -->"

    markdown_text = _BLOCK_EQUATION_PATTERN.sub(block_repl, markdown_text)

    def inline_repl(match: re.Match[str]) -> str:
        nonlocal counter
        expression = match.group(1).strip()
        eq_id = f"eq_{counter:03d}"
        token = f"[[eq:{eq_id}]]"
        equations.append(
            ExtractedEquation(
                id=eq_id,
                latex_or_text=expression,
                display_mode="inline",
                page=None,
                markdown_anchor=token,
            )
        )
        counter += 1
        return f"${expression}$<!-- {token} -->"

    markdown_text = _INLINE_EQUATION_PATTERN.sub(inline_repl, markdown_text)
    return markdown_text, equations


def _annotate_markdown(
    markdown_text: str,
    images: list[ExtractedImage],
    tables: list[ExtractedTable],
) -> tuple[str, list[ExtractedEquation]]:
    markdown_text, equations = _annotate_equations(markdown_text)
    lines = ["", "## Artifact References", ""]
    for item in images:
        lines.append(f"- [[img:{item.id}]] -> Attached Image {item.id}")
    for item in tables:
        lines.append(f"- [[tbl:{item.id}]] -> Attached Table {item.id}")
    lines.append("")
    markdown_text = markdown_text.rstrip() + "\n" + "\n".join(lines)
    return markdown_text, equations


# ---------------------------------------------------------------------------
# Backend class
# ---------------------------------------------------------------------------


class DoclingBackend(OCRBackend):
    """OCR backend powered by Docling.

    Owns the full extraction pipeline: PDF parsing via DocumentConverter,
    hybrid chunking, image/table/equation extraction, markdown annotation,
    and manifest generation.
    """

    def extract(self, source_pdf_path: str) -> ExtractionResult:
        _disable_hf_symlink_usage_on_windows()
        source = Path(source_pdf_path)
        if not source.exists():
            raise FileNotFoundError(f"Input PDF not found: {source_pdf_path}")

        start_time = time.time()
        doc_id = _slugify(source.stem) or "document"
        print(f"Starting extraction for {source.name} (ID: {doc_id})")

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=_build_pdf_pipeline_options()
                ),
            }
        )

        print("Parsing and converting PDF into Docling Document...")
        conv_res = converter.convert(source)
        document = conv_res.document
        _infer_heading_depth_from_numbering(document)

        print("Chunking document in memory...")
        source_chunks = _extract_chunks(document)

        print("Extracting markdown with referenced images...")
        markdown_text = document.export_to_markdown(
            image_mode=ImageRefMode.REFERENCED
        )

        print("Extracting image metadata and tables from memory...")
        images = _build_image_metadata(document)
        tables = _extract_artifacts(
            document=document,
            attr_name="tables",
            prefix="tbl",
            kind="table",
            expected_type=ExtractedTable,
        )

        print("Annotating equations...")
        markdown_text, equations = _annotate_markdown(
            markdown_text=markdown_text,
            images=images,
            tables=tables,
        )

        print("Building references...")
        references = build_artifact_references(
            ("image", "img", images),
            ("table", "tbl", tables),
            ("equation", "eq", equations),
        )

        print("Injecting artifact tokens into chunks...")
        for eq in equations:
            for chunk in source_chunks:
                if eq.latex_or_text and eq.latex_or_text in chunk.text:
                    chunk.text = chunk.text.replace(
                        eq.latex_or_text,
                        f"{eq.latex_or_text} {eq.markdown_anchor}",
                    )
                    chunk.contextualized_text = chunk.contextualized_text.replace(
                        eq.latex_or_text,
                        f"{eq.latex_or_text} {eq.markdown_anchor}",
                    )

        for img in images:
            injected = False
            token = f"[[img:{img.id}]]"
            if img.caption:
                for chunk in source_chunks:
                    if chunk.captions and any(
                        img.caption in c for c in chunk.captions
                    ):
                        chunk.text += f"\n\n{token}"
                        chunk.contextualized_text += f"\n\n{token}"
                        injected = True
                        break
                if not injected:
                    for chunk in source_chunks:
                        if img.caption in chunk.text:
                            chunk.text += f"\n\n{token}"
                            chunk.contextualized_text += f"\n\n{token}"
                            injected = True
                            break
            if not injected and source_chunks:
                source_chunks[-1].text += f"\n\n{token}"
                source_chunks[-1].contextualized_text += f"\n\n{token}"

        for tbl in tables:
            injected = False
            token = f"[[tbl:{tbl.id}]]"
            if tbl.title:
                for chunk in source_chunks:
                    if chunk.captions and any(
                        tbl.title in c for c in chunk.captions
                    ):
                        chunk.text += f"\n\n{token}"
                        chunk.contextualized_text += f"\n\n{token}"
                        injected = True
                        break
                if not injected:
                    for chunk in source_chunks:
                        if tbl.title in chunk.text:
                            chunk.text += f"\n\n{token}"
                            chunk.contextualized_text += f"\n\n{token}"
                            injected = True
                            break
            if not injected and source_chunks:
                source_chunks[-1].text += f"\n\n{token}"
                source_chunks[-1].contextualized_text += f"\n\n{token}"

        print("Generating manifest...")
        manifest = ExtractionManifest(
            doc_id=doc_id,
            source_pdf_path=str(source),
            markdown_path="",
            images=images,
            tables=tables,
            equations=equations,
            references=references,
        )

        print("Validating references...")
        _verify_references_in_markdown(
            markdown_text=markdown_text, manifest=manifest
        )

        print(f"[{time.time() - start_time:.2f}s] Extraction complete.")
        return ExtractionResult(
            source_chunks=source_chunks,
            manifest_json=manifest,
            image_count=len(images),
            table_count=len(tables),
            equation_count=len(equations),
            chunk_count=len(source_chunks),
        )
