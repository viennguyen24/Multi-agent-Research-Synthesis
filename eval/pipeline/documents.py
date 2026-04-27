from __future__ import annotations

import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import Callable

from eval.config import EvalConfig
from eval.pipeline.assets import download_file, hashed_filename, resolve_paper_download_url
from eval.pipeline.common import get_benchmark_loader, init_eval_storage, utc_now
from eval.schema import Task


DocumentProcessor = Callable[[list[str], str | None], object]


def register_benchmark_tasks(
    config: EvalConfig,
    benchmark_id: str,
    preserve_processed: bool = True,
) -> list[Task]:
    manifest_path = config.benchmark_manifests[benchmark_id]
    loader = get_benchmark_loader(benchmark_id)
    load_result = loader.load(manifest_path)
    created_at = utc_now()
    registered_tasks: list[Task] = []
    with init_eval_storage(config) as db:
        for task in load_result.tasks:
            existing = db.get_task(task.benchmark_id, task.suite_id, task.task_id)
            registered_task = task
            if preserve_processed and existing and _is_processed_task(existing, config.research_db_path):
                registered_task = _merge_processed_task(task, existing)
            elif existing and existing.source_document_paths:
                registered_task = _merge_downloaded_task(task, existing)
            db.register_task(registered_task, created_at)
            registered_tasks.append(registered_task)
    return registered_tasks


def build_documents(
    config: EvalConfig,
    benchmark_id: str,
    suite_id: str | None = None,
    task_ids: list[str] | None = None,
    processor: DocumentProcessor | None = None,
    force_process: bool = False,
) -> list[str]:
    download_documents(
        config=config,
        benchmark_id=benchmark_id,
        suite_id=suite_id,
        task_ids=task_ids,
        force_download=force_process,
    )
    return process_documents(
        config=config,
        benchmark_id=benchmark_id,
        suite_id=suite_id,
        task_ids=task_ids,
        processor=processor,
        force_process=force_process,
    )


def download_documents(
    config: EvalConfig,
    benchmark_id: str,
    suite_id: str | None = None,
    task_ids: list[str] | None = None,
    force_download: bool = False,
) -> list[str]:
    register_benchmark_tasks(config, benchmark_id, preserve_processed=not force_download)
    with init_eval_storage(config) as db:
        tasks = db.list_tasks(benchmark_id=benchmark_id, suite_id=suite_id, task_ids=task_ids)

    downloaded_paths: OrderedDict[str, None] = OrderedDict()
    created_at = utc_now()
    with init_eval_storage(config) as db:
        for task in tasks:
            if not task.source_paper_url:
                raise ValueError(f"Task {task.task_id} is missing source_paper_url")
            if not force_download and _has_downloaded_assets(task):
                downloaded_paths[task.source_document_paths[0]] = None
                continue

            paper_url = resolve_paper_download_url(task.source_paper_url)
            local_paper = download_file(
                paper_url,
                config.paths.source_docs_dir / hashed_filename(task.task_id, paper_url, ".pdf"),
            )
            local_reference: str | None = None
            slides_url = task.metadata.get("slides_url")
            if slides_url:
                local_reference = str(
                    download_file(
                        slides_url,
                        config.paths.reference_decks_dir / hashed_filename(task.task_id, slides_url, ".pdf"),
                    )
                )

            downloaded_paths[str(local_paper)] = None
            db.register_task(
                Task(
                    benchmark_id=task.benchmark_id,
                    suite_id=task.suite_id,
                    task_id=task.task_id,
                    query=task.query,
                    source_document_id=task.task_id if force_download else task.source_document_id,
                    source_document_paths=[str(local_paper)],
                    source_paper_url=task.source_paper_url,
                    key_facts=task.key_facts,
                    raw_reference_deck_path=local_reference,
                    metadata={
                        **task.metadata,
                        "local_paper_path": str(local_paper),
                        "local_reference_deck_path": local_reference,
                    },
                ),
                created_at,
            )
    return list(downloaded_paths.keys())


def process_documents(
    config: EvalConfig,
    benchmark_id: str,
    suite_id: str | None = None,
    task_ids: list[str] | None = None,
    processor: DocumentProcessor | None = None,
    force_process: bool = False,
) -> list[str]:
    register_benchmark_tasks(config, benchmark_id, preserve_processed=not force_process)
    with init_eval_storage(config) as db:
        tasks = db.list_tasks(benchmark_id=benchmark_id, suite_id=suite_id, task_ids=task_ids)

    tasks_to_process = [task for task in tasks if force_process or not _is_processed_task(task, config.research_db_path)]
    unique_paths: OrderedDict[str, None] = OrderedDict()
    for task in tasks_to_process:
        if not task.source_document_paths:
            raise ValueError(
                f"Task {task.task_id} is missing downloaded source_document_paths. Run download_documents first."
            )
        unique_paths[task.source_document_paths[0]] = None
    if not unique_paths:
        return []

    runner = processor or _default_document_processor
    result = runner(list(unique_paths.keys()), str(config.paths.llm_config_path))
    processed_by_source = {
        item["source_path"]: item for item in getattr(result, "processed_documents", [])
    }

    created_at = utc_now()
    with init_eval_storage(config) as db:
        for task in tasks_to_process:
            local_paper = task.source_document_paths[0]
            processed = processed_by_source.get(local_paper)
            if processed is None:
                raise RuntimeError(f"Document processing did not produce a doc_id for {local_paper}")
            db.register_task(
                Task(
                    benchmark_id=task.benchmark_id,
                    suite_id=task.suite_id,
                    task_id=task.task_id,
                    query=task.query,
                    source_document_id=processed["doc_id"],
                    source_document_paths=[local_paper],
                    source_paper_url=task.source_paper_url,
                    key_facts=task.key_facts,
                    raw_reference_deck_path=task.raw_reference_deck_path,
                    metadata={
                        **task.metadata,
                        "paper_title": processed["paper_title"],
                    },
                ),
                created_at,
            )
    return list(unique_paths.keys())


def _default_document_processor(pdf_paths: list[str], llm_config_path: str | None) -> object:
    from eval.graph_runner import process_documents as run_document_processor

    return run_document_processor(
        pdf_paths=pdf_paths,
        llm_config_path=llm_config_path,
        database_path=None,
    )


def _merge_processed_task(manifest_task: Task, existing_task: Task) -> Task:
    return Task(
        benchmark_id=manifest_task.benchmark_id,
        suite_id=manifest_task.suite_id,
        task_id=manifest_task.task_id,
        query=manifest_task.query,
        source_document_id=existing_task.source_document_id,
        source_document_paths=list(existing_task.source_document_paths),
        source_paper_url=manifest_task.source_paper_url,
        key_facts=list(existing_task.key_facts or manifest_task.key_facts),
        raw_reference_deck_path=existing_task.raw_reference_deck_path,
        metadata={**manifest_task.metadata, **existing_task.metadata},
    )


def _merge_downloaded_task(manifest_task: Task, existing_task: Task) -> Task:
    return Task(
        benchmark_id=manifest_task.benchmark_id,
        suite_id=manifest_task.suite_id,
        task_id=manifest_task.task_id,
        query=manifest_task.query,
        source_document_id=existing_task.source_document_id,
        source_document_paths=list(existing_task.source_document_paths),
        source_paper_url=manifest_task.source_paper_url,
        key_facts=list(existing_task.key_facts or manifest_task.key_facts),
        raw_reference_deck_path=existing_task.raw_reference_deck_path,
        metadata={**manifest_task.metadata, **existing_task.metadata},
    )


def _has_downloaded_assets(task: Task) -> bool:
    if not task.source_document_paths:
        return False
    local_paper = Path(task.source_document_paths[0])
    if not local_paper.exists():
        return False
    if not task.raw_reference_deck_path:
        return True
    return Path(task.raw_reference_deck_path).exists()


def _is_processed_task(task: Task, research_db_path: Path) -> bool:
    if task.source_document_id == task.task_id:
        return False
    if not _has_downloaded_assets(task):
        return False
    return _document_exists(research_db_path, task.source_document_id)


def _document_exists(research_db_path: Path, document_id: str) -> bool:
    with sqlite3.connect(research_db_path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'documents' LIMIT 1"
        ).fetchone()
        if row is None:
            return False
        document_row = conn.execute("SELECT 1 FROM documents WHERE id = ? LIMIT 1", (document_id,)).fetchone()
    return document_row is not None
