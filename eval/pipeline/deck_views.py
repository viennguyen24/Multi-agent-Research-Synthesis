from __future__ import annotations

import uuid
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from eval.config import EvalConfig
from eval.pipeline.common import init_eval_storage, utc_now, write_json_artifact
from eval.schema import DeckView, DeckViewSlide, DeckViewTextBlock


XML_NAMESPACES = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}


def build_deck_views(
    config: EvalConfig,
    benchmark_id: str,
    suite_id: str | None = None,
    task_ids: list[str] | None = None,
    extraction_version: str = "pptx_xml_v1",
) -> list[DeckView]:
    outputs: list[DeckView] = []
    with init_eval_storage(config) as db:
        transcripts = db.list_transcripts(benchmark_id=benchmark_id, suite_id=suite_id, task_ids=task_ids)
        references = db.list_reference_rows(benchmark_id=benchmark_id, suite_id=suite_id, task_ids=task_ids)

    for transcript in transcripts:
        if not transcript.final_deck_path:
            continue
        outputs.append(
            _persist_deck_view(
                config=config,
                source_kind="generated",
                source_id=transcript.transcript_id,
                source_path=transcript.final_deck_path,
                suite_id=transcript.suite_id,
                transcript_id=transcript.transcript_id,
                extraction_version=extraction_version,
            )
        )
    for row in references:
        outputs.append(
            _persist_deck_view(
                config=config,
                source_kind="reference",
                source_id=row["reference_id"],
                source_path=row["raw_reference_deck_path"],
                suite_id=row["suite_id"],
                transcript_id=None,
                extraction_version=extraction_version,
            )
        )
    return outputs


def _persist_deck_view(
    config: EvalConfig,
    source_kind: str,
    source_id: str,
    source_path: str,
    suite_id: str | None,
    transcript_id: str | None,
    extraction_version: str,
) -> DeckView:
    slides = _extract_slides_from_pptx(source_path)
    deck_view = DeckView(
        deck_view_id=str(uuid.uuid4()),
        source_kind=source_kind,
        source_id=source_id,
        source_path=source_path,
        extraction_version=extraction_version,
        suite_id=suite_id,
        transcript_id=transcript_id,
        slides=slides,
    )
    created_at = utc_now()
    artifact_path = write_json_artifact(
        config.paths.deck_views_dir / f"{deck_view.deck_view_id}.json",
        deck_view.to_dict(),
    )
    with init_eval_storage(config) as db:
        db.insert_deck_view(
            deck_view_id=deck_view.deck_view_id,
            source_kind=deck_view.source_kind,
            source_id=deck_view.source_id,
            source_path=deck_view.source_path,
            extraction_version=deck_view.extraction_version,
            suite_id=deck_view.suite_id,
            transcript_id=deck_view.transcript_id,
            artifact_path=artifact_path,
            created_at=created_at,
        )
        db.index_artifact(
            artifact_id=deck_view.deck_view_id,
            kind="deck_view",
            owner_type=source_kind,
            owner_id=source_id,
            path=artifact_path,
            metadata={"source_path": source_path},
            created_at=created_at,
        )
    return deck_view


def _extract_slides_from_pptx(source_path: str) -> list[DeckViewSlide]:
    path = Path(source_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Deck file not found: {path}")
    slides: list[DeckViewSlide] = []
    with zipfile.ZipFile(path, "r") as archive:
        slide_names = sorted(
            name for name in archive.namelist()
            if name.startswith("ppt/slides/slide") and name.endswith(".xml")
        )
        for slide_index, slide_name in enumerate(slide_names):
            xml_text = archive.read(slide_name)
            root = ElementTree.fromstring(xml_text)
            blocks: list[DeckViewTextBlock] = []
            for block_index, node in enumerate(root.findall(".//a:t", XML_NAMESPACES)):
                text = (node.text or "").strip()
                if not text:
                    continue
                blocks.append(
                    DeckViewTextBlock(
                        block_index=block_index,
                        text=text,
                        provenance=f"{slide_name}#text[{block_index}]",
                    )
                )
            combined_text = "\n".join(block.text for block in blocks)
            slides.append(
                DeckViewSlide(
                    slide_index=slide_index,
                    source_ref=slide_name,
                    text_blocks=blocks,
                    combined_text=combined_text,
                )
            )
    return slides
