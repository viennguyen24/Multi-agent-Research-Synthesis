from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ImageRefMode

EQUATION_TOKEN_PATTERN = re.compile(r"\[\[eq:eq_\d{3}\]\]")
INLINE_EQUATION_PATTERN = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")
BLOCK_EQUATION_PATTERN = re.compile(r"\$\$(.+?)\$\$", flags=re.DOTALL)
SECTION_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)\s")


def _disable_hf_symlink_usage_on_windows() -> None:
    """Force huggingface_hub to copy files instead of symlinking on Windows.  You get errors otherwise if you're running on Windows not in admin mode."""
    if os.name != "nt":
        return

    # Silence repeated warning noise in non-admin Windows sessions.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    try:
        import huggingface_hub.file_download as hf_file_download
    except Exception:
        return

    hf_file_download._are_symlinks_supported_in_dir.clear()

    def _always_false(_cache_dir: str | Path | None = None) -> bool:
        return False

    hf_file_download.are_symlinks_supported = _always_false


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _fix_heading_levels(document: Any) -> None:
    """Infer proper heading depth from section numbering (e.g. '3.2.1' -> level 3).

    Docling's PDF backend currently detects all section headers at level 1,
    which prevents the hierarchical chunker from building a breadcrumb trail.
    We parse the leading numbering pattern to recover the true depth.

    Unnumbered headings that appear inside a deeper numbered section (e.g. a
    figure sub-title misidentified as a section header) are demoted so they
    nest below the last numbered heading instead of resetting the hierarchy.
    """
    headers = [
        item for item in document.texts
        if "section_header" in str(getattr(item, "label", ""))
    ]

    has_numbered = False
    for item in headers:
        text = (getattr(item, "text", "") or "").strip()
        m = SECTION_NUMBER_RE.match(text)
        if m:
            item.level = m.group(1).count(".") + 1
            has_numbered = True

    if not has_numbered:
        return

    last_numbered_level = 0
    for item in headers:
        text = (getattr(item, "text", "") or "").strip()
        if SECTION_NUMBER_RE.match(text):
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


def _annotate_equations(markdown_text: str) -> tuple[str, list[dict[str, Any]]]:
    equations: list[dict[str, Any]] = []
    counter = 1

    def block_repl(match: re.Match[str]) -> str:
        nonlocal counter
        expression = match.group(1).strip()
        eq_id = f"eq_{counter:03d}"
        token = f"[[eq:{eq_id}]]"
        equations.append(
            {
                "id": eq_id,
                "latex_or_text": expression,
                "display_mode": "block",
                "page": None,
                "markdown_anchor": token,
            }
        )
        counter += 1
        return f"$${expression}$$\n<!-- {token} -->"

    markdown_text = BLOCK_EQUATION_PATTERN.sub(block_repl, markdown_text)

    def inline_repl(match: re.Match[str]) -> str:
        nonlocal counter
        expression = match.group(1).strip()
        eq_id = f"eq_{counter:03d}"
        token = f"[[eq:{eq_id}]]"
        equations.append(
            {
                "id": eq_id,
                "latex_or_text": expression,
                "display_mode": "inline",
                "page": None,
                "markdown_anchor": token,
            }
        )
        counter += 1
        return f"${expression}$<!-- {token} -->"

    markdown_text = INLINE_EQUATION_PATTERN.sub(inline_repl, markdown_text)
    return markdown_text, equations


def _append_reference_section(
    markdown_text: str,
    images: list[dict[str, Any]],
    tables: list[dict[str, Any]],
) -> str:
    lines = ["", "## Artifact References", ""]
    for item in images:
        lines.append(f"- [[img:{item['id']}]] -> {item['path']}")
    for item in tables:
        lines.append(f"- [[tbl:{item['id']}]] -> {item['path']}")
    lines.append("")
    return markdown_text.rstrip() + "\n" + "\n".join(lines)


def _validate_references(markdown_text: str, manifest: dict[str, Any]) -> None:
    markdown_tokens = set(re.findall(r"\[\[(?:img|tbl|eq):[a-z0-9_]+\]\]", markdown_text))
    if not markdown_tokens:
        raise RuntimeError("No artifact reference tokens were written into markdown.")

    references = manifest.get("references", [])
    manifest_tokens = {str(ref.get("token", "")).strip() for ref in references if ref.get("token")}

    missing_from_manifest = sorted(token for token in markdown_tokens if token not in manifest_tokens)
    if missing_from_manifest:
        raise RuntimeError(f"Markdown tokens missing from manifest: {missing_from_manifest}")

    equation_anchors = {
        str(eq.get("markdown_anchor", "")).strip()
        for eq in manifest.get("equations", [])
        if eq.get("markdown_anchor")
    }
    missing_equation_anchors = sorted(anchor for anchor in equation_anchors if anchor not in markdown_tokens)
    if missing_equation_anchors:
        raise RuntimeError(f"Equation anchors missing from markdown: {missing_equation_anchors}")


def extract_multimodal_pdf_artifacts(source_pdf_path: str) -> dict[str, Any]:
    _disable_hf_symlink_usage_on_windows()
    source = Path(source_pdf_path)
    if not source.exists():
        raise FileNotFoundError(f"Input PDF not found: {source_pdf_path}")

    doc_id = _slugify(source.stem) or "document"
    artifact_root = Path("artifacts") / doc_id
    images_dir = artifact_root / "images"
    tables_dir = artifact_root / "tables"
    markdown_path = artifact_root / "document.md"
    equations_path = artifact_root / "equations.jsonl"
    manifest_path = artifact_root / "manifest.json"

    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    artifact_root.mkdir(parents=True)
    images_dir.mkdir(parents=True)
    tables_dir.mkdir(parents=True)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_picture_images = True
    pipeline_options.do_formula_enrichment = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    conv_res = converter.convert(source)
    document = conv_res.document

    _fix_heading_levels(document)

    chunker = HybridChunker()
    source_chunks: list[dict[str, Any]] = []
    for chunk in chunker.chunk(dl_doc=document):
        source_chunks.append(
            {
                "text": chunk.text,
                "contextualized_text": chunker.contextualize(chunk),
                "headings": list(chunk.meta.headings) if chunk.meta.headings else [],
                "captions": list(chunk.meta.captions) if chunk.meta.captions else [],
            }
        )

    chunks_path = artifact_root / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as fp:
        for chunk_dict in source_chunks:
            fp.write(json.dumps(chunk_dict, ensure_ascii=False) + "\n")

    document.save_as_markdown(markdown_path, image_mode=ImageRefMode.PLACEHOLDER)
    markdown_text = markdown_path.read_text(encoding="utf-8")

    images: list[dict[str, Any]] = []
    for idx, picture in enumerate(getattr(document, "pictures", []), start=1):
        image = picture.get_image(document)
        if image is None:
            continue
        image_id = f"img_{idx:03d}"
        rel_path = Path("images") / f"{image_id}.png"
        out_path = artifact_root / rel_path
        image.save(out_path, "PNG")
        images.append(
            {
                "id": image_id,
                "path": str(rel_path).replace("\\", "/"),
                "page": _extract_page_no(picture),
                "caption": _extract_caption(picture),
            }
        )

    tables: list[dict[str, Any]] = []
    for idx, table in enumerate(getattr(document, "tables", []), start=1):
        table_id = f"tbl_{idx:03d}"
        rel_path = Path("tables") / f"{table_id}.html"
        out_path = artifact_root / rel_path
        out_path.write_text(table.export_to_html(doc=document), encoding="utf-8")
        tables.append(
            {
                "id": table_id,
                "path": str(rel_path).replace("\\", "/"),
                "page": _extract_page_no(table),
                "title": f"Table {idx}",
            }
        )

    markdown_text, equations = _annotate_equations(markdown_text)
    markdown_text = _append_reference_section(markdown_text, images=images, tables=tables)
    markdown_path.write_text(markdown_text, encoding="utf-8")

    with equations_path.open("w", encoding="utf-8") as fp:
        for equation in equations:
            fp.write(json.dumps(equation, ensure_ascii=True) + "\n")

    references: list[dict[str, str]] = []
    for image in images:
        references.append({"token": f"[[img:{image['id']}]]", "item_id": image["id"], "kind": "image"})
    for table in tables:
        references.append({"token": f"[[tbl:{table['id']}]]", "item_id": table["id"], "kind": "table"})
    for equation in equations:
        references.append(
            {
                "token": equation["markdown_anchor"],
                "item_id": equation["id"],
                "kind": "equation",
            }
        )

    manifest = {
        "doc_id": doc_id,
        "source_pdf_path": str(source),
        "markdown_path": str(markdown_path),
        "images": images,
        "tables": tables,
        "equations": equations,
        "references": references,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")

    _validate_references(markdown_text=markdown_text, manifest=manifest)

    return {
        "source_chunks": source_chunks,
        "manifest_json": manifest,
        "image_count": len(images),
        "table_count": len(tables),
        "equation_count": len(equations),
        "chunk_count": len(source_chunks),
    }
