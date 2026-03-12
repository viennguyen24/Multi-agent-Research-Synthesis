from dataclasses import dataclass
from typing import Literal, TypeVar

@dataclass
class ExtractedImage:
    id: str
    path: str
    page: int | None
    caption: str

@dataclass
class ExtractedTable:
    id: str
    path: str
    page: int | None
    title: str

@dataclass
class ExtractedEquation:
    id: str
    latex_or_text: str
    display_mode: Literal["block", "inline"]
    page: int | None
    markdown_anchor: str

@dataclass
class ExtractedChunk:
    text: str
    contextualized_text: str
    headings: list[str]
    captions: list[str]

@dataclass
class ArtifactReference:
    token: str
    item_id: str
    kind: Literal["image", "table", "equation"]

@dataclass
class ExtractionManifest:
    doc_id: str
    source_pdf_path: str
    markdown_path: str
    images: list[ExtractedImage]
    tables: list[ExtractedTable]
    equations: list[ExtractedEquation]
    references: list[ArtifactReference]

@dataclass
class ExtractionResult:
    source_chunks: list[ExtractedChunk]
    manifest_json: ExtractionManifest
    image_count: int
    table_count: int
    equation_count: int
    chunk_count: int

T = TypeVar("T", ExtractedImage, ExtractedTable)
