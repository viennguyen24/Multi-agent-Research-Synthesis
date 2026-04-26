from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class EvalPaths:
    root_dir: Path
    eval_db: Path
    artifacts_dir: Path
    transcripts_dir: Path
    deck_views_dir: Path
    metric_results_dir: Path
    reports_dir: Path
    runtime_dbs_dir: Path
    llm_config_path: Path


@dataclass(slots=True)
class EvalConfig:
    paths: EvalPaths
    benchmark_manifests: dict[str, Path]
    research_db_path: Path
    output_dir: Path


def _resolve_path(root_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (root_dir / path).resolve()


def load_eval_config(config_path: str | Path) -> EvalConfig:
    config_file = Path(config_path).expanduser().resolve()
    root_dir = config_file.parent
    with config_file.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    storage = data.get("storage") or {}
    artifacts = storage.get("artifacts") or {}
    benchmarks = data.get("benchmarks") or {}
    runtime = data.get("runtime") or {}

    paths = EvalPaths(
        root_dir=root_dir,
        eval_db=_resolve_path(root_dir, storage.get("eval_db", "data/eval.db")),
        artifacts_dir=_resolve_path(root_dir, storage.get("artifacts_dir", "artifacts")),
        transcripts_dir=_resolve_path(root_dir, artifacts.get("transcripts", "artifacts/transcripts")),
        deck_views_dir=_resolve_path(root_dir, artifacts.get("deck_views", "artifacts/deck_views")),
        metric_results_dir=_resolve_path(root_dir, artifacts.get("metric_results", "artifacts/metric_results")),
        reports_dir=_resolve_path(root_dir, artifacts.get("reports", "artifacts/reports")),
        runtime_dbs_dir=_resolve_path(root_dir, artifacts.get("runtime_dbs", "artifacts/runtime_dbs")),
        llm_config_path=_resolve_path(root_dir, data.get("llm_config", "llm.config.yaml")),
    )
    benchmark_manifests = {
        benchmark_id: _resolve_path(root_dir, str(raw_path))
        for benchmark_id, raw_path in benchmarks.items()
    }

    return EvalConfig(
        paths=paths,
        benchmark_manifests=benchmark_manifests,
        research_db_path=_resolve_path(root_dir, runtime.get("research_db", "data/research.db")),
        output_dir=_resolve_path(root_dir, runtime.get("output_dir", "output")),
    )


def ensure_eval_directories(config: EvalConfig) -> None:
    for path in (
        config.paths.eval_db.parent,
        config.paths.artifacts_dir,
        config.paths.transcripts_dir,
        config.paths.deck_views_dir,
        config.paths.metric_results_dir,
        config.paths.reports_dir,
        config.paths.runtime_dbs_dir,
        config.output_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def dump_yaml(data: dict[str, Any], output_path: str | Path) -> None:
    with Path(output_path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)

