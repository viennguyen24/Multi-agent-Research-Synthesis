from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from eval.metrics.benchmark.deck_bench.deck_metrics import (
    deck_coherence_llm_metric,
    deck_faithfulness_metric,
    deck_fidelity_metric,
    dtw_sequence_similarity_metric,
    transition_similarity_metric,
)
from eval.metrics.benchmark.deck_bench.slide_metrics import (
    EmbedderProtocol,
    slide_content_quality_llm_metric,
    slide_faithfulness_metric,
    slide_semantic_similarity_metric,
    slide_text_similarity_metric,
)
from eval.schema import DeckView, MetricResult, Transcript


@dataclass(slots=True)
class DeckBenchMetricSuite:
    embedder: EmbedderProtocol
    llm_config_path: str | None
    secondary_output_dir: Path

    def grade_transcript(
        self,
        transcript: Transcript,
        generated_deck: DeckView,
        reference_deck: DeckView,
        source_chunks: list[str],
    ) -> list[MetricResult]:
        created_at = datetime.now(timezone.utc).isoformat()
        results: list[MetricResult] = []
        slide_count = min(len(generated_deck.slides), len(reference_deck.slides))
        for index in range(slide_count):
            generated_slide = generated_deck.slides[index]
            reference_slide = reference_deck.slides[index]
            results.append(self._make_result(
                transcript=transcript,
                metric_id="slide_text_similarity",
                subject_type="generated_slide",
                subject_id=f"{transcript.transcript_id}:slide:{generated_slide.slide_index}",
                scalar_value=slide_text_similarity_metric(generated_slide, reference_slide),
                created_at=created_at,
            ))
            results.append(self._make_result(
                transcript=transcript,
                metric_id="slide_faithfulness",
                subject_type="generated_slide",
                subject_id=f"{transcript.transcript_id}:slide:{generated_slide.slide_index}",
                scalar_value=slide_faithfulness_metric(generated_slide, source_chunks),
                created_at=created_at,
            ))
            results.append(self._make_result(
                transcript=transcript,
                metric_id="slide_semantic_similarity",
                subject_type="generated_slide",
                subject_id=f"{transcript.transcript_id}:slide:{generated_slide.slide_index}",
                scalar_value=slide_semantic_similarity_metric(generated_slide, reference_slide, self.embedder),
                created_at=created_at,
            ))
            if self.llm_config_path is not None:
                score, reason = slide_content_quality_llm_metric(generated_slide, self.llm_config_path)
                results.append(self._make_result(
                    transcript=transcript,
                    metric_id="slide_content_quality",
                    subject_type="generated_slide",
                    subject_id=f"{transcript.transcript_id}:slide:{generated_slide.slide_index}",
                    scalar_value=score,
                    reason=reason,
                    created_at=created_at,
                ))

        results.append(self._make_result(
            transcript=transcript,
            metric_id="deck_faithfulness",
            subject_type="generated_deck",
            subject_id=transcript.transcript_id,
            scalar_value=deck_faithfulness_metric(generated_deck, source_chunks, self.embedder),
            created_at=created_at,
        ))
        results.append(self._make_result(
            transcript=transcript,
            metric_id="deck_fidelity",
            subject_type="generated_deck",
            subject_id=transcript.transcript_id,
            scalar_value=deck_fidelity_metric(generated_deck, reference_deck, self.embedder),
            created_at=created_at,
        ))
        results.append(self._make_result(
            transcript=transcript,
            metric_id="transition_similarity",
            subject_type="generated_deck",
            subject_id=transcript.transcript_id,
            scalar_value=transition_similarity_metric(generated_deck, reference_deck),
            created_at=created_at,
        ))
        dtw_score, dtw_path = dtw_sequence_similarity_metric(generated_deck, reference_deck)
        dtw_artifact = self.secondary_output_dir / f"{transcript.transcript_id}_dtw.json"
        dtw_artifact.write_text(json.dumps({"path": dtw_path}, indent=2), encoding="utf-8")
        results.append(self._make_result(
            transcript=transcript,
            metric_id="dtw_sequence_similarity",
            subject_type="generated_deck",
            subject_id=transcript.transcript_id,
            scalar_value=dtw_score,
            artifact_path=str(dtw_artifact),
            metadata={"alignment_path": dtw_path},
            created_at=created_at,
        ))
        if self.llm_config_path is not None:
            score, reason = deck_coherence_llm_metric(generated_deck, self.llm_config_path)
            results.append(self._make_result(
                transcript=transcript,
                metric_id="deck_coherence",
                subject_type="generated_deck",
                subject_id=transcript.transcript_id,
                scalar_value=score,
                reason=reason,
                created_at=created_at,
            ))
        return results

    def _make_result(
        self,
        transcript: Transcript,
        metric_id: str,
        subject_type: str,
        subject_id: str,
        scalar_value: float,
        created_at: str,
        artifact_path: str | None = None,
        metadata: dict | None = None,
        reason: str | None = None,
    ) -> MetricResult:
        del created_at
        return MetricResult(
            metric_result_id=str(uuid.uuid4()),
            transcript_id=transcript.transcript_id,
            suite_id=transcript.suite_id,
            trial_index=transcript.trial_index,
            variant_id=transcript.variant_id,
            graph_version=transcript.graph_version,
            doc_pipeline_version=transcript.doc_pipeline_version,
            benchmark_id=transcript.benchmark_id,
            metric_id=metric_id,
            grader_id="deck_bench_v1",
            subject_type=subject_type,
            subject_id=subject_id,
            status="success",
            scalar_value=scalar_value,
            reason=reason,
            artifact_path=artifact_path,
            metadata=metadata or {},
        )
