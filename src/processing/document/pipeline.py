import json
import time
from dataclasses import asdict
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode

from .schema import ExtractedImage, ExtractedTable, ExtractionManifest, ExtractionResult
from .utils import (
    _disable_hf_symlink_usage_on_windows,
    _infer_heading_depth_from_numbering,
    _slugify,
    _verify_references_in_markdown,
    annotate_markdown,
    build_artifact_references,
    build_image_metadata_from_document,
    build_pdf_pipeline_options,
    extract_chunks,
    extract_artifacts,
)


def extract_multimodal_pdf_artifacts(source_pdf_path: str) -> ExtractionResult:
    """End-to-end pipeline that parses a PDF into semantic chunks of text units, markdown, images, tables, and equations.
    
    Artifacts are stored on disk with images saved as PNGs, tables as HTML
    and text components chunked in a JSONL file. The function returns an `ExtractionResult` containing 
    the parsed chunks and a manifest tracking all generated assets and their markdown reference tokens.
    """
    _disable_hf_symlink_usage_on_windows()
    source = Path(source_pdf_path)
    if not source.exists():
        raise FileNotFoundError(f"Input PDF not found: {source_pdf_path}")
    
    start_time = time.time()
    doc_id = _slugify(source.stem) or "document"
    print(f"Starting extraction for {source.name} (ID: {doc_id})")
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=build_pdf_pipeline_options()),
        }
    )
    
    print("Parsing and converting PDF into Docling Document...")
    conv_res = converter.convert(source)
    document = conv_res.document
    _infer_heading_depth_from_numbering(document)

    print("Chunking document in memory...")
    source_chunks = extract_chunks(document)

    print("Extracting markdown with referenced images...")
    # Using ImageRefMode.REFERENCED will keep the unique IDs in the markdown.
    # Since we are not writing to disk, it won't actually export PNG files.
    markdown_text = document.export_to_markdown(image_mode=ImageRefMode.REFERENCED)

    print("Extracting image metadata and tables from memory...")
    images = build_image_metadata_from_document(
        document=document,
    )
    tables = extract_artifacts(
        document=document,
        attr_name="tables",
        prefix="tbl",
        kind="table",
        expected_type=ExtractedTable
    )

    print("Annotating equations...")
    markdown_text, equations = annotate_markdown(
        markdown_text=markdown_text,
        images=images,
        tables=tables,
    )

    print("Building references...")
    references = build_artifact_references(
        ("image", "img", images),
        ("table", "tbl", tables),
        ("equation", "eq", equations)
    )

    print("Injecting artifact tokens into chunks...")
    for eq in equations:
        for chunk in source_chunks:
            if eq.latex_or_text and eq.latex_or_text in chunk.text:
                chunk.text = chunk.text.replace(eq.latex_or_text, f"{eq.latex_or_text} {eq.markdown_anchor}")
                chunk.contextualized_text = chunk.contextualized_text.replace(eq.latex_or_text, f"{eq.latex_or_text} {eq.markdown_anchor}")

    for img in images:
        injected = False
        token = f"[[img:{img.id}]]"
        if img.caption:
            for chunk in source_chunks:
                # 1. match in chunk captions metadata
                if chunk.captions and any(img.caption in c for c in chunk.captions):
                    chunk.text += f"\n\n{token}"
                    chunk.contextualized_text += f"\n\n{token}"
                    injected = True
                    break
            if not injected:
                # 2. match in chunk text directly
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
                if chunk.captions and any(tbl.title in c for c in chunk.captions):
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
        markdown_path="",  # Markdown is stored dynamically now.
        images=images,
        tables=tables,
        equations=equations,
        references=references,
    )

    print("Validating references...")
    _verify_references_in_markdown(markdown_text=markdown_text, manifest=manifest)

    print(f"[{time.time() - start_time:.2f}s] Extraction complete.")
    return ExtractionResult(
        source_chunks=source_chunks,
        manifest_json=manifest,
        image_count=len(images),
        table_count=len(tables),
        equation_count=len(equations),
        chunk_count=len(source_chunks),
    )
