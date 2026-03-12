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
    annotate_and_save_markdown,
    build_artifact_paths,
    build_artifact_references,
    build_image_metadata_from_saved,
    build_pdf_pipeline_options,
    extract_and_save_chunks,
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
    paths = build_artifact_paths(doc_id)
    artifact_root = paths["artifact_root"]
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

    print("Chunking document...")
    source_chunks = extract_and_save_chunks(document, paths["chunks_path"])

    print("Saving markdown with referenced images...")
    document.save_as_markdown(
        paths["markdown_path"],
        image_mode=ImageRefMode.REFERENCED,
        artifacts_dir=Path("images"),
    )
    markdown_text = paths["markdown_path"].read_text(encoding="utf-8")

    print("Building image metadata and extracting tables...")
    images = build_image_metadata_from_saved(
        document=document,
        images_dir=paths["images_dir"],
    )
    tables = extract_artifacts(
        document=document,
        artifact_root=artifact_root,
        attr_name="tables",
        prefix="tbl",
        kind="table",
        expected_type=ExtractedTable
    )

    print("Annotating equations...")
    markdown_text, equations = annotate_and_save_markdown(
        markdown_text=markdown_text,
        images=images,
        tables=tables,
        markdown_path=paths["markdown_path"],
        equations_path=paths["equations_path"]
    )

    print("Building references...")
    references = build_artifact_references(
        ("image", "img", images),
        ("table", "tbl", tables),
        ("equation", "eq", equations)
    )

    print("Saving manifest...")
    manifest = ExtractionManifest(
        doc_id=doc_id,
        source_pdf_path=str(source),
        markdown_path=str(paths["markdown_path"]),
        images=images,
        tables=tables,
        equations=equations,
        references=references,
    )
    paths["manifest_path"].write_text(json.dumps(asdict(manifest), indent=2, ensure_ascii=True), encoding="utf-8")

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
