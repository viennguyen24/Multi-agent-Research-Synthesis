from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from eval.db.schema import CREATE_STATEMENTS
from eval.schema import MetricResult, Task, Transcript


INSERT_BENCHMARK_TASK_SQL = """
INSERT OR REPLACE INTO benchmark_tasks (
    benchmark_id, suite_id, task_id, query, source_document_id,
    source_document_paths_json, key_facts_json, raw_reference_deck_path,
    metadata_json, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

INSERT_BENCHMARK_REFERENCE_SQL = """
INSERT OR REPLACE INTO benchmark_references (
    benchmark_id, suite_id, task_id, reference_id, raw_reference_deck_path,
    metadata_json, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?)
"""

SELECT_TASKS_BASE_SQL = """
SELECT *
FROM benchmark_tasks
WHERE benchmark_id = ?
"""

SELECT_TASK_SQL = """
SELECT *
FROM benchmark_tasks
WHERE benchmark_id = ? AND suite_id = ? AND task_id = ?
"""

SELECT_REFERENCES_BASE_SQL = """
SELECT *
FROM benchmark_references
WHERE benchmark_id = ?
"""

INSERT_TRANSCRIPT_SQL = """
INSERT INTO transcripts (
    transcript_id, benchmark_id, suite_id, task_id, trial_index,
    variant_id, graph_version, doc_pipeline_version, status, created_at,
    finished_at, session_id, query, source_document_id, final_deck_path,
    transcript_artifact_path, final_state_artifact_path, node_events_artifact_path,
    debug_artifact_path, error_text
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

SELECT_TRANSCRIPTS_BASE_SQL = "SELECT * FROM transcripts WHERE 1 = 1"

SELECT_DECK_VIEW_BY_SOURCE_SQL = """
SELECT *
FROM deck_views
WHERE source_kind = ? AND source_id = ?
ORDER BY created_at DESC
LIMIT 1
"""

INSERT_DECK_VIEW_SQL = """
INSERT INTO deck_views (
    deck_view_id, source_kind, source_id, source_path, extraction_version,
    suite_id, transcript_id, artifact_path, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

SELECT_DECK_VIEWS_BASE_SQL = "SELECT * FROM deck_views WHERE 1 = 1"

INSERT_METRIC_RESULT_SQL = """
INSERT INTO metric_results (
    metric_result_id, transcript_id, suite_id, trial_index, variant_id,
    graph_version, doc_pipeline_version, benchmark_id, metric_id, grader_id,
    subject_type, subject_id, status, scalar_value, pass_fail, reason,
    artifact_path, metadata_json, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

SELECT_METRIC_RESULTS_BASE_SQL = "SELECT * FROM metric_results WHERE 1 = 1"

INSERT_ARTIFACT_INDEX_SQL = """
INSERT OR REPLACE INTO artifact_index (
    artifact_id, kind, owner_type, owner_id, path, metadata_json, created_at
) VALUES (?, ?, ?, ?, ?, ?, ?)
"""


class EvalDatabase:
    """Small SQLite abstraction for eval metadata and artifact indexes."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self.setup()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "EvalDatabase":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def setup(self) -> None:
        with self._conn:
            for statement in CREATE_STATEMENTS:
                self._conn.execute(statement)

    def register_task(self, task: Task, created_at: str) -> None:
        with self._conn:
            self._conn.execute(
                INSERT_BENCHMARK_TASK_SQL,
                (
                    task.benchmark_id,
                    task.suite_id,
                    task.task_id,
                    task.query,
                    task.source_document_id,
                    json.dumps(task.source_document_paths),
                    json.dumps(task.key_facts),
                    task.raw_reference_deck_path,
                    json.dumps(task.metadata),
                    created_at,
                ),
            )
            if task.raw_reference_deck_path:
                self._conn.execute(
                    INSERT_BENCHMARK_REFERENCE_SQL,
                    (
                        task.benchmark_id,
                        task.suite_id,
                        task.task_id,
                        f"{task.benchmark_id}:{task.suite_id}:{task.task_id}",
                        task.raw_reference_deck_path,
                        json.dumps(task.metadata),
                        created_at,
                    ),
                )

    def list_tasks(
        self,
        benchmark_id: str,
        suite_id: str | None = None,
        task_ids: list[str] | None = None,
    ) -> list[Task]:
        query = SELECT_TASKS_BASE_SQL
        params: list[Any] = [benchmark_id]
        if suite_id:
            query += " AND suite_id = ?"
            params.append(suite_id)
        if task_ids:
            placeholders = ",".join("?" for _ in task_ids)
            query += f" AND task_id IN ({placeholders})"
            params.extend(task_ids)
        query += " ORDER BY suite_id, task_id"
        return [self._task_from_row(row) for row in self._conn.execute(query, params).fetchall()]

    def get_task(self, benchmark_id: str, suite_id: str, task_id: str) -> Task | None:
        row = self._conn.execute(SELECT_TASK_SQL, (benchmark_id, suite_id, task_id)).fetchone()
        return self._task_from_row(row) if row else None

    def list_reference_rows(
        self,
        benchmark_id: str,
        suite_id: str | None = None,
        task_ids: list[str] | None = None,
    ) -> list[sqlite3.Row]:
        query = SELECT_REFERENCES_BASE_SQL
        params: list[Any] = [benchmark_id]
        if suite_id:
            query += " AND suite_id = ?"
            params.append(suite_id)
        if task_ids:
            placeholders = ",".join("?" for _ in task_ids)
            query += f" AND task_id IN ({placeholders})"
            params.extend(task_ids)
        query += " ORDER BY suite_id, task_id"
        return self._conn.execute(query, params).fetchall()

    def insert_transcript(self, transcript: Transcript, created_at: str) -> None:
        with self._conn:
            self._conn.execute(
                INSERT_TRANSCRIPT_SQL,
                (
                    transcript.transcript_id,
                    transcript.benchmark_id,
                    transcript.suite_id,
                    transcript.task_id,
                    transcript.trial_index,
                    transcript.variant_id,
                    transcript.graph_version,
                    transcript.doc_pipeline_version,
                    transcript.status,
                    created_at,
                    transcript.finished_at,
                    transcript.session_id,
                    transcript.query,
                    transcript.source_document_id,
                    transcript.final_deck_path,
                    transcript.transcript_artifact_path,
                    transcript.final_state_artifact_path,
                    transcript.node_events_artifact_path,
                    transcript.debug_artifact_path,
                    transcript.error_text,
                ),
            )

    def list_transcripts(
        self,
        benchmark_id: str | None = None,
        suite_id: str | None = None,
        task_ids: list[str] | None = None,
    ) -> list[Transcript]:
        query = SELECT_TRANSCRIPTS_BASE_SQL
        params: list[Any] = []
        if benchmark_id:
            query += " AND benchmark_id = ?"
            params.append(benchmark_id)
        if suite_id:
            query += " AND suite_id = ?"
            params.append(suite_id)
        if task_ids:
            placeholders = ",".join("?" for _ in task_ids)
            query += f" AND task_id IN ({placeholders})"
            params.extend(task_ids)
        query += " ORDER BY created_at, trial_index"
        return [self._transcript_from_row(row) for row in self._conn.execute(query, params).fetchall()]

    def get_deck_view_by_source(self, source_kind: str, source_id: str) -> sqlite3.Row | None:
        return self._conn.execute(SELECT_DECK_VIEW_BY_SOURCE_SQL, (source_kind, source_id)).fetchone()

    def insert_deck_view(self, deck_view_id: str, source_kind: str, source_id: str, source_path: str, extraction_version: str, suite_id: str | None, transcript_id: str | None, artifact_path: str, created_at: str) -> None:
        with self._conn:
            self._conn.execute(
                INSERT_DECK_VIEW_SQL,
                (
                    deck_view_id,
                    source_kind,
                    source_id,
                    source_path,
                    extraction_version,
                    suite_id,
                    transcript_id,
                    artifact_path,
                    created_at,
                ),
            )

    def list_deck_views(self, source_kind: str | None = None, suite_id: str | None = None) -> list[sqlite3.Row]:
        query = SELECT_DECK_VIEWS_BASE_SQL
        params: list[Any] = []
        if source_kind:
            query += " AND source_kind = ?"
            params.append(source_kind)
        if suite_id:
            query += " AND suite_id = ?"
            params.append(suite_id)
        query += " ORDER BY created_at"
        return self._conn.execute(query, params).fetchall()

    def insert_metric_result(self, result: MetricResult, created_at: str) -> None:
        with self._conn:
            self._conn.execute(
                INSERT_METRIC_RESULT_SQL,
                (
                    result.metric_result_id,
                    result.transcript_id,
                    result.suite_id,
                    result.trial_index,
                    result.variant_id,
                    result.graph_version,
                    result.doc_pipeline_version,
                    result.benchmark_id,
                    result.metric_id,
                    result.grader_id,
                    result.subject_type,
                    result.subject_id,
                    result.status,
                    result.scalar_value,
                    None if result.pass_fail is None else int(result.pass_fail),
                    result.reason,
                    result.artifact_path,
                    json.dumps(result.metadata),
                    created_at,
                ),
            )

    def list_metric_results(self, benchmark_id: str | None = None, suite_id: str | None = None) -> list[sqlite3.Row]:
        query = SELECT_METRIC_RESULTS_BASE_SQL
        params: list[Any] = []
        if benchmark_id:
            query += " AND benchmark_id = ?"
            params.append(benchmark_id)
        if suite_id:
            query += " AND suite_id = ?"
            params.append(suite_id)
        query += " ORDER BY created_at, metric_id, subject_id"
        return self._conn.execute(query, params).fetchall()

    def index_artifact(
        self,
        artifact_id: str,
        kind: str,
        owner_type: str,
        owner_id: str,
        path: str,
        metadata: dict[str, Any],
        created_at: str,
    ) -> None:
        with self._conn:
            self._conn.execute(
                INSERT_ARTIFACT_INDEX_SQL,
                (
                    artifact_id,
                    kind,
                    owner_type,
                    owner_id,
                    path,
                    json.dumps(metadata),
                    created_at,
                ),
            )

    @staticmethod
    def _task_from_row(row: sqlite3.Row) -> Task:
        return Task(
            benchmark_id=row["benchmark_id"],
            suite_id=row["suite_id"],
            task_id=row["task_id"],
            query=row["query"],
            source_document_id=row["source_document_id"],
            source_document_paths=json.loads(row["source_document_paths_json"]),
            source_paper_url=json.loads(row["metadata_json"]).get("paper_url"),
            key_facts=json.loads(row["key_facts_json"]),
            raw_reference_deck_path=row["raw_reference_deck_path"],
            metadata=json.loads(row["metadata_json"]),
        )

    @staticmethod
    def _transcript_from_row(row: sqlite3.Row) -> Transcript:
        return Transcript(
            transcript_id=row["transcript_id"],
            benchmark_id=row["benchmark_id"],
            suite_id=row["suite_id"],
            task_id=row["task_id"],
            trial_index=row["trial_index"],
            variant_id=row["variant_id"],
            graph_version=row["graph_version"],
            doc_pipeline_version=row["doc_pipeline_version"],
            status=row["status"],
            created_at=row["created_at"],
            finished_at=row["finished_at"],
            session_id=row["session_id"],
            query=row["query"],
            source_document_id=row["source_document_id"],
            final_deck_path=row["final_deck_path"],
            transcript_artifact_path=row["transcript_artifact_path"],
            final_state_artifact_path=row["final_state_artifact_path"],
            node_events_artifact_path=row["node_events_artifact_path"],
            debug_artifact_path=row["debug_artifact_path"],
            error_text=row["error_text"],
        )
