from __future__ import annotations


CREATE_BENCHMARK_TASKS_TABLE = """
CREATE TABLE IF NOT EXISTS benchmark_tasks (
    benchmark_id TEXT NOT NULL,
    suite_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    query TEXT NOT NULL,
    source_document_id TEXT NOT NULL,
    source_document_paths_json TEXT NOT NULL,
    key_facts_json TEXT NOT NULL,
    raw_reference_deck_path TEXT,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (benchmark_id, suite_id, task_id)
)
"""


CREATE_BENCHMARK_REFERENCES_TABLE = """
CREATE TABLE IF NOT EXISTS benchmark_references (
    benchmark_id TEXT NOT NULL,
    suite_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    reference_id TEXT NOT NULL,
    raw_reference_deck_path TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (reference_id)
)
"""


CREATE_TRANSCRIPTS_TABLE = """
CREATE TABLE IF NOT EXISTS transcripts (
    transcript_id TEXT PRIMARY KEY,
    benchmark_id TEXT NOT NULL,
    suite_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    trial_index INTEGER NOT NULL,
    variant_id TEXT NOT NULL,
    graph_version TEXT NOT NULL,
    doc_pipeline_version TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,
    source_document_id TEXT NOT NULL,
    final_deck_path TEXT,
    transcript_artifact_path TEXT,
    final_state_artifact_path TEXT,
    node_events_artifact_path TEXT,
    debug_artifact_path TEXT,
    error_text TEXT
)
"""


CREATE_DECK_VIEWS_TABLE = """
CREATE TABLE IF NOT EXISTS deck_views (
    deck_view_id TEXT PRIMARY KEY,
    source_kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_path TEXT NOT NULL,
    extraction_version TEXT NOT NULL,
    suite_id TEXT,
    transcript_id TEXT,
    artifact_path TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


CREATE_METRIC_RESULTS_TABLE = """
CREATE TABLE IF NOT EXISTS metric_results (
    metric_result_id TEXT PRIMARY KEY,
    transcript_id TEXT NOT NULL,
    suite_id TEXT NOT NULL,
    trial_index INTEGER NOT NULL,
    variant_id TEXT NOT NULL,
    graph_version TEXT NOT NULL,
    doc_pipeline_version TEXT NOT NULL,
    benchmark_id TEXT NOT NULL,
    metric_id TEXT NOT NULL,
    grader_id TEXT NOT NULL,
    subject_type TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    status TEXT NOT NULL,
    scalar_value REAL,
    pass_fail INTEGER,
    reason TEXT,
    artifact_path TEXT,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


CREATE_ARTIFACT_INDEX_TABLE = """
CREATE TABLE IF NOT EXISTS artifact_index (
    artifact_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    owner_type TEXT NOT NULL,
    owner_id TEXT NOT NULL,
    path TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


CREATE_STATEMENTS = [
    CREATE_BENCHMARK_TASKS_TABLE,
    CREATE_BENCHMARK_REFERENCES_TABLE,
    CREATE_TRANSCRIPTS_TABLE,
    CREATE_DECK_VIEWS_TABLE,
    CREATE_METRIC_RESULTS_TABLE,
    CREATE_ARTIFACT_INDEX_TABLE,
]

