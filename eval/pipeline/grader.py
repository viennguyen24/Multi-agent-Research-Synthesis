from __future__ import annotations

from pathlib import Path

from eval.config import EvalConfig
from eval.pipeline.common import init_eval_storage, utc_now, write_json_artifact
from eval.schema import DeckView


def run_grader(
    config: EvalConfig,
    benchmark_id: str,
    suite_id: str | None = None,
    research_db_factory=None,
) -> int:
    with init_eval_storage(config) as db:
        transcripts = db.list_transcripts(benchmark_id=benchmark_id, suite_id=suite_id)
        results_written = 0
        llm_config_path = str(config.paths.llm_config_path) if config.paths.llm_config_path.exists() else None
        deckbench_suite = None
        presentbench_suite = None
        if research_db_factory is None:
            from src.memory.research.config import StorageConfig
            from src.memory.research.database import ResearchDatabase

            research_db = ResearchDatabase(StorageConfig(db_path=config.research_db_path))
        else:
            research_db = research_db_factory(config.research_db_path)
        try:
            for transcript in transcripts:
                document = research_db.load_document(transcript.source_document_id)
                if document is None:
                    raise FileNotFoundError(
                        f"Missing processed source document {transcript.source_document_id}"
                    )
                task = db.get_task(transcript.benchmark_id, transcript.suite_id, transcript.task_id)
                if task is None:
                    raise FileNotFoundError(
                        f"Missing benchmark task for transcript {transcript.transcript_id}"
                    )

                if benchmark_id == "presentbench":
                    if presentbench_suite is None:
                        from eval.metrics.benchmark.presentbench.metric_suite import PresentBenchMetricSuite

                        presentbench_suite = PresentBenchMetricSuite(
                            llm_config_path=llm_config_path,
                            secondary_output_dir=config.paths.metric_results_dir,
                        )
                    if not transcript.final_deck_path:
                        raise FileNotFoundError(
                            f"Missing generated deck path for transcript {transcript.transcript_id}"
                        )
                    source_text = (document.markdown or "").strip()
                    if not source_text:
                        source_text = "\n\n".join(
                            chunk.contextualized_text or chunk.text for chunk in document.source_chunks
                        )
                    results = presentbench_suite.grade_transcript(
                        transcript=transcript,
                        task=task,
                        candidate_deck_path=Path(transcript.final_deck_path),
                        source_text=source_text,
                    )
                else:
                    if deckbench_suite is None:
                        from eval.metrics.benchmark.deck_bench.metric_suite import DeckBenchMetricSuite
                        from src.processing.embedder.provider import get_text_embedder

                        deckbench_suite = DeckBenchMetricSuite(
                            embedder=get_text_embedder(),
                            llm_config_path=llm_config_path,
                            secondary_output_dir=config.paths.metric_results_dir,
                        )
                    generated_row = db.get_deck_view_by_source("generated", transcript.transcript_id)
                    if generated_row is None:
                        raise FileNotFoundError(
                            f"Missing generated deck view for transcript {transcript.transcript_id}"
                        )
                    reference_id = f"{transcript.benchmark_id}:{transcript.suite_id}:{transcript.task_id}"
                    reference_row = db.get_deck_view_by_source("reference", reference_id)
                    if reference_row is None:
                        raise FileNotFoundError(f"Missing reference deck view for task {transcript.task_id}")
                    generated_deck = _load_deck_view(Path(generated_row["artifact_path"]))
                    reference_deck = _load_deck_view(Path(reference_row["artifact_path"]))
                    source_chunks = [chunk.contextualized_text or chunk.text for chunk in document.source_chunks]
                    results = deckbench_suite.grade_transcript(transcript, generated_deck, reference_deck, source_chunks)
                created_at = utc_now()
                for result in results:
                    if result.artifact_path:
                        artifact_path = result.artifact_path
                    else:
                        artifact_path = write_json_artifact(
                            config.paths.metric_results_dir / f"{result.metric_result_id}.json",
                            result.to_dict(),
                        )
                        result.artifact_path = artifact_path
                    db.insert_metric_result(result, created_at)
                    db.index_artifact(
                        artifact_id=result.metric_result_id,
                        kind="metric_result",
                        owner_type="transcript",
                        owner_id=transcript.transcript_id,
                        path=result.artifact_path or artifact_path,
                        metadata={"metric_id": result.metric_id},
                        created_at=created_at,
                    )
                    results_written += 1
        finally:
            research_db.close()
    return results_written


def _load_deck_view(path: Path) -> DeckView:
    import json

    return DeckView.from_dict(json.loads(path.read_text(encoding="utf-8")))
