from __future__ import annotations

from collections import OrderedDict
from typing import Callable

from eval.config import EvalConfig
from eval.pipeline.common import get_benchmark_loader, init_eval_storage, utc_now
from eval.schema import Task


def register_benchmark_tasks(
    config: EvalConfig,
    benchmark_id: str,
) -> list[Task]:
    manifest_path = config.benchmark_manifests[benchmark_id]
    loader = get_benchmark_loader(benchmark_id)
    load_result = loader.load(manifest_path)
    created_at = utc_now()
    with init_eval_storage(config) as db:
        for task in load_result.tasks:
            db.register_task(task, created_at)
    return load_result.tasks


def build_documents(
    config: EvalConfig,
    benchmark_id: str,
    suite_id: str | None = None,
    task_ids: list[str] | None = None,
    processor: Callable[[list[str], str | None], object] | None = None,
) -> list[str]:
    register_benchmark_tasks(config, benchmark_id)
    with init_eval_storage(config) as db:
        tasks = db.list_tasks(benchmark_id=benchmark_id, suite_id=suite_id, task_ids=task_ids)

    unique_paths: OrderedDict[str, None] = OrderedDict()
    for task in tasks:
        for path in task.source_document_paths:
            unique_paths[path] = None
    if not unique_paths:
        return []

    runner = processor or _default_document_processor
    runner(list(unique_paths.keys()), str(config.paths.llm_config_path))
    return list(unique_paths.keys())


def _default_document_processor(pdf_paths: list[str], llm_config_path: str | None) -> object:
    from eval.graph_runner import process_documents

    return process_documents(
        pdf_paths=pdf_paths,
        llm_config_path=llm_config_path,
        database_path=None,
    )
