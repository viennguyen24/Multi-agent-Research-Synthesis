from __future__ import annotations

import uuid
from typing import Any, Callable

from eval.config import EvalConfig
from eval.pipeline.common import (
    clone_research_db,
    init_eval_storage,
    utc_now,
    write_json_artifact,
)
from eval.pipeline.documents import register_benchmark_tasks
from eval.schema import Task, Transcript


RunnerCallable = Callable[..., Any]


def run_harness(
    config: EvalConfig,
    benchmark_id: str,
    variant_id: str,
    graph_version: str,
    doc_pipeline_version: str,
    trial_count: int,
    suite_id: str | None = None,
    task_ids: list[str] | None = None,
    runner: RunnerCallable | None = None,
) -> list[Transcript]:
    register_benchmark_tasks(config, benchmark_id)
    with init_eval_storage(config) as db:
        tasks = db.list_tasks(benchmark_id=benchmark_id, suite_id=suite_id, task_ids=task_ids)

    if runner is None:
        from eval.graph_runner import run_existing_documents

        execute = run_existing_documents
    else:
        execute = runner
    transcripts: list[Transcript] = []
    for task in tasks:
        for trial_index in range(trial_count):
            transcripts.append(
                _run_single_trial(
                    config=config,
                    task=task,
                    variant_id=variant_id,
                    graph_version=graph_version,
                    doc_pipeline_version=doc_pipeline_version,
                    trial_index=trial_index,
                    runner=execute,
                )
            )
    return transcripts


def _run_single_trial(
    config: EvalConfig,
    task: Task,
    variant_id: str,
    graph_version: str,
    doc_pipeline_version: str,
    trial_index: int,
    runner: RunnerCallable,
) -> Transcript:
    transcript_id = str(uuid.uuid4())
    created_at = utc_now()
    runtime_db_path = clone_research_db(config.research_db_path, config.paths.runtime_dbs_dir)
    transcript_dir = config.paths.transcripts_dir / transcript_id
    transcript_dir.mkdir(parents=True, exist_ok=True)

    status = "failed"
    finished_at = None
    error_text = None
    final_deck_path = None
    session_id = ""
    final_state_artifact_path = None
    node_events_artifact_path = None
    debug_artifact_path = None
    transcript_artifact_path = None

    try:
        run_result = runner(
            query=task.query,
            doc_ids=[task.source_document_id],
            paper_titles=[task.metadata.get("paper_title", task.task_id)],
            llm_config_path=str(config.paths.llm_config_path),
            output_dir=str(config.output_dir),
            database_path=str(runtime_db_path),
            existing_docs_only=True,
            clear_run_artifacts=True,
        )
        finished_at = utc_now()
        status = run_result.status
        error_text = run_result.error_text
        final_deck_path = run_result.pptx_path
        session_id = run_result.session_id

        final_state_artifact_path = write_json_artifact(
            transcript_dir / "final_state.json",
            run_result.final_state,
        )
        node_events_artifact_path = write_json_artifact(
            transcript_dir / "node_events.json",
            [event.to_dict() for event in run_result.node_events],
        )
        debug_artifact_path = write_json_artifact(
            transcript_dir / "debug.json",
            {
                "runtime_db_path": str(runtime_db_path),
                "warnings": run_result.final_warnings,
            },
        )
    except Exception as exc:
        finished_at = utc_now()
        session_id = ""
        error_text = str(exc)

    transcript = Transcript(
        transcript_id=transcript_id,
        benchmark_id=task.benchmark_id,
        suite_id=task.suite_id,
        task_id=task.task_id,
        trial_index=trial_index,
        variant_id=variant_id,
        graph_version=graph_version,
        doc_pipeline_version=doc_pipeline_version,
        status=status,
        created_at=created_at,
        finished_at=finished_at,
        session_id=session_id,
        query=task.query,
        source_document_id=task.source_document_id,
        final_deck_path=final_deck_path,
        final_state_artifact_path=final_state_artifact_path,
        node_events_artifact_path=node_events_artifact_path,
        debug_artifact_path=debug_artifact_path,
        error_text=error_text,
    )
    transcript_artifact_path = write_json_artifact(transcript_dir / "transcript.json", transcript.to_dict())
    transcript.transcript_artifact_path = transcript_artifact_path
    with init_eval_storage(config) as db:
        db.insert_transcript(transcript, created_at)
        db.index_artifact(
            artifact_id=transcript_id,
            kind="transcript",
            owner_type="transcript",
            owner_id=transcript_id,
            path=transcript_artifact_path,
            metadata={"status": status},
            created_at=created_at,
        )
    return transcript
