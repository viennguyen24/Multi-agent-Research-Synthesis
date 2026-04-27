from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


TaskMetadata = dict[str, Any]


@dataclass(slots=True)
class Task:
    benchmark_id: str
    suite_id: str
    task_id: str
    query: str

    source_document_id: str
    source_document_paths: list[str] = field(default_factory=list)
    source_paper_url: str | None = None
    
    key_facts: list[str] = field(default_factory=list)
    raw_reference_deck_path: str | None = None
    metadata: TaskMetadata = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class NodeEvent:
    transcript_id: str
    sequence_index: int
    event_type: str
    timestamp: str
    node_name: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Transcript:
    transcript_id: str
    benchmark_id: str
    suite_id: str
    task_id: str
    trial_index: int
    variant_id: str
    graph_version: str
    doc_pipeline_version: str

    status: Literal["success", "failed", "partial"]
    created_at: str
    finished_at: str | None
    session_id: str
    
    query: str
    source_document_id: str
    final_deck_path: str | None = None
    transcript_artifact_path: str | None = None
    final_state_artifact_path: str | None = None
    node_events_artifact_path: str | None = None
    debug_artifact_path: str | None = None
    error_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DeckViewTextBlock:
    block_index: int
    text: str
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DeckViewTextBlock":
        return cls(
            block_index=payload["block_index"],
            text=payload["text"],
            provenance=payload["provenance"],
        )


@dataclass(slots=True)
class DeckViewSlide:
    slide_index: int
    source_ref: str
    text_blocks: list[DeckViewTextBlock]
    combined_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "slide_index": self.slide_index,
            "source_ref": self.source_ref,
            "text_blocks": [block.to_dict() for block in self.text_blocks],
            "combined_text": self.combined_text,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DeckViewSlide":
        return cls(
            slide_index=payload["slide_index"],
            source_ref=payload["source_ref"],
            text_blocks=[DeckViewTextBlock.from_dict(block) for block in payload["text_blocks"]],
            combined_text=payload["combined_text"],
        )


@dataclass(slots=True)
class DeckView:
    deck_view_id: str
    source_kind: Literal["generated", "reference"]
    source_id: str
    source_path: str
    extraction_version: str
    suite_id: str | None
    transcript_id: str | None = None
    slides: list[DeckViewSlide] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "deck_view_id": self.deck_view_id,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "source_path": self.source_path,
            "extraction_version": self.extraction_version,
            "suite_id": self.suite_id,
            "transcript_id": self.transcript_id,
            "slides": [slide.to_dict() for slide in self.slides],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DeckView":
        return cls(
            deck_view_id=payload["deck_view_id"],
            source_kind=payload["source_kind"],
            source_id=payload["source_id"],
            source_path=payload["source_path"],
            extraction_version=payload["extraction_version"],
            suite_id=payload.get("suite_id"),
            transcript_id=payload.get("transcript_id"),
            slides=[DeckViewSlide.from_dict(slide) for slide in payload.get("slides", [])],
        )


@dataclass(slots=True)
class MetricResult:
    metric_result_id: str
    transcript_id: str
    suite_id: str
    trial_index: int
    variant_id: str
    graph_version: str
    doc_pipeline_version: str
    benchmark_id: str
    metric_id: str
    grader_id: str

    subject_type: str
    subject_id: str
    status: Literal["success", "failed", "skipped"]
    scalar_value: float | None = None
    pass_fail: bool | None = None
    reason: str | None = None

    artifact_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
