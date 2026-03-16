"""Shared helpers that depend only on the schema, not on any OCR library."""
import re
from typing import Literal

from .schema import (
    ArtifactReference,
    ExtractedEquation,
    ExtractedImage,
    ExtractedTable,
    ExtractionManifest,
)


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _verify_references_in_markdown(
    markdown_text: str, manifest: ExtractionManifest
) -> None:
    """Ensure all artifacts in the manifest have corresponding reference tokens in the markdown."""
    markdown_tokens = set(
        re.findall(r"\[\[(?:img|tbl|eq):[a-z0-9_]+\]\]", markdown_text)
    )
    if not markdown_tokens:
        raise RuntimeError("No artifact reference tokens were written into markdown.")

    manifest_tokens = {
        str(ref.token).strip() for ref in manifest.references if ref.token
    }

    missing_from_manifest = sorted(
        token for token in markdown_tokens if token not in manifest_tokens
    )
    if missing_from_manifest:
        raise RuntimeError(
            f"Markdown tokens missing from manifest: {missing_from_manifest}"
        )

    equation_anchors = {
        str(eq.markdown_anchor).strip()
        for eq in manifest.equations
        if eq.markdown_anchor
    }
    missing_equation_anchors = sorted(
        anchor for anchor in equation_anchors if anchor not in markdown_tokens
    )
    if missing_equation_anchors:
        raise RuntimeError(
            f"Equation anchors missing from markdown: {missing_equation_anchors}"
        )


def build_artifact_references(
    *artifact_groups: tuple[
        Literal["image", "table", "equation"],
        str,
        list[ExtractedImage | ExtractedTable | ExtractedEquation],
    ]
) -> list[ArtifactReference]:
    references: list[ArtifactReference] = []
    for kind, prefix, items in artifact_groups:
        for item in items:
            token = getattr(item, "markdown_anchor", f"[[{prefix}:{item.id}]]")
            references.append(
                ArtifactReference(token=token, item_id=item.id, kind=kind)
            )
    return references
