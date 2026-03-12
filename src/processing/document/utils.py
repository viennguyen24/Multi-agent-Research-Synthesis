import json
import os
import re
import shutil
import psutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, TypeVar

from docling.chunking import HybridChunker
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions
from docling_core.types.doc import DoclingDocument

from .schema import (
    ArtifactReference,
    ExtractedChunk,
    ExtractedEquation,
    ExtractedImage,
    ExtractedTable,
    ExtractionManifest,
    T,
)

EQUATION_TOKEN_PATTERN = re.compile(r"\[\[eq:eq_\d{3}\]\]")
INLINE_EQUATION_PATTERN = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")
BLOCK_EQUATION_PATTERN = re.compile(r"\$\$(.+?)\$\$", flags=re.DOTALL)
SECTION_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)\s")


def _disable_hf_symlink_usage_on_windows() -> None:
    """Force huggingface_hub to copy files instead of symlinking on Windows."""
    if os.name != "nt":
        return

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


def _infer_heading_depth_from_numbering(document: DoclingDocument) -> None:
    """Infer proper heading depth from section numbering."""
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

    markdown_text = BLOCK_EQUATION_PATTERN.sub(block_repl, markdown_text)

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

    markdown_text = INLINE_EQUATION_PATTERN.sub(inline_repl, markdown_text)
    return markdown_text, equations


def _append_reference_section(
    markdown_text: str,
    images: list[ExtractedImage],
    tables: list[ExtractedTable],
) -> str:
    lines = ["", "## Artifact References", ""]
    for item in images:
        lines.append(f"- [[img:{item.id}]] -> {item.path}")
    for item in tables:
        lines.append(f"- [[tbl:{item.id}]] -> {item.path}")
    lines.append("")
    return markdown_text.rstrip() + "\n" + "\n".join(lines)


def _verify_references_in_markdown(markdown_text: str, manifest: ExtractionManifest) -> None:
    """Ensure all artifacts in the manifest have corresponding reference tokens in the markdown."""
    markdown_tokens = set(re.findall(r"\[\[(?:img|tbl|eq):[a-z0-9_]+\]\]", markdown_text))
    if not markdown_tokens:
        raise RuntimeError("No artifact reference tokens were written into markdown.")

    manifest_tokens = {str(ref.token).strip() for ref in manifest.references if ref.token}

    missing_from_manifest = sorted(token for token in markdown_tokens if token not in manifest_tokens)
    if missing_from_manifest:
        raise RuntimeError(f"Markdown tokens missing from manifest: {missing_from_manifest}")

    equation_anchors = {
        str(eq.markdown_anchor).strip()
        for eq in manifest.equations
        if eq.markdown_anchor
    }
    missing_equation_anchors = sorted(anchor for anchor in equation_anchors if anchor not in markdown_tokens)
    if missing_equation_anchors:
        raise RuntimeError(f"Equation anchors missing from markdown: {missing_equation_anchors}")


def build_artifact_paths(doc_id: str) -> dict[str, Path]:
    artifact_root = Path("artifacts") / doc_id
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    
    images_dir = artifact_root / "images"
    tables_dir = artifact_root / "tables"
    
    artifact_root.mkdir(parents=True)
    images_dir.mkdir(parents=True)
    tables_dir.mkdir(parents=True)
    
    return {
        "artifact_root": artifact_root,
        "images_dir": images_dir,
        "tables_dir": tables_dir,
        "markdown_path": artifact_root / "document.md",
        "equations_path": artifact_root / "equations.jsonl",
        "manifest_path": artifact_root / "manifest.json",
        "chunks_path": artifact_root / "chunks.jsonl",
    }


def build_pdf_pipeline_options() -> PdfPipelineOptions:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_picture_images = True
    pipeline_options.do_formula_enrichment = True
    # Set the number of threads to use for the accelerator to number of physical cores
    pipeline_options.accelerator_options = AcceleratorOptions(num_threads=psutil.cpu_count(logical=False), device="cpu")
    return pipeline_options


def write_artifact(
    artifact_root: Path,
    item: Any,
    item_id: str,
    kind: Literal["image", "table"],
    document: DoclingDocument
) -> ExtractedImage | ExtractedTable | None:
    if kind == "image":
        image = item.get_image(document)
        if image is None:
            return None
        rel_path = Path("images") / f"{item_id}.png"
        out_path = artifact_root / rel_path
        image.save(out_path, "PNG")
        return ExtractedImage(
            id=item_id,
            path=str(rel_path).replace("\\", "/"),
            page=_extract_page_no(item),
            caption=_extract_caption(item),
        )
    elif kind == "table":
        rel_path = Path("tables") / f"{item_id}.html"
        out_path = artifact_root / rel_path
        out_path.write_text(item.export_to_html(doc=document), encoding="utf-8")
        idx = int(item_id.split("_")[-1])
        return ExtractedTable(
            id=item_id,
            path=str(rel_path).replace("\\", "/"),
            page=_extract_page_no(item),
            title=f"Table {idx}",
        )
    return None


def extract_and_save_chunks(document: DoclingDocument, chunks_path: Path) -> list[ExtractedChunk]:
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

    with chunks_path.open("w", encoding="utf-8") as fp:
        for chunk in source_chunks:
            fp.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
            
    return source_chunks


def extract_artifacts(
    document: DoclingDocument,
    artifact_root: Path,
    attr_name: str,
    prefix: str,
    kind: Literal["image", "table"],
    expected_type: type[T]
) -> list[T]:
    extracted_items: list[T] = []
    items = getattr(document, attr_name, [])
    for idx, item in enumerate(items, start=1):
        item_id = f"{prefix}_{idx:03d}"
        res = write_artifact(artifact_root, item, item_id, kind, document)
        if res and isinstance(res, expected_type):
            extracted_items.append(res)
    return extracted_items


def build_artifact_references(
    *artifact_groups: tuple[Literal["image", "table", "equation"], str, list[ExtractedImage | ExtractedTable | ExtractedEquation]]
) -> list[ArtifactReference]:
    references: list[ArtifactReference] = []
    for kind, prefix, items in artifact_groups:
        for item in items:
            token = getattr(item, "markdown_anchor", f"[[{prefix}:{item.id}]]")
            references.append(ArtifactReference(token=token, item_id=item.id, kind=kind))
    return references


def annotate_and_save_markdown(
    markdown_text: str,
    images: list[ExtractedImage],
    tables: list[ExtractedTable],
    markdown_path: Path,
    equations_path: Path
) -> tuple[str, list[ExtractedEquation]]:
    markdown_text, equations = _annotate_equations(markdown_text)
    markdown_text = _append_reference_section(markdown_text, images=images, tables=tables)
    markdown_path.write_text(markdown_text, encoding="utf-8")

    with equations_path.open("w", encoding="utf-8") as fp:
        for equation in equations:
            fp.write(json.dumps(asdict(equation), ensure_ascii=True) + "\n")
            
    return markdown_text, equations
