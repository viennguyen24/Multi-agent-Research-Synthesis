from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.reports.base import BenchmarkLoader
from eval.config import EvalConfig, ensure_eval_directories
from eval.db import EvalDatabase
from eval.metrics.benchmark.deck_bench.loader import DeckBenchLoader
from eval.metrics.benchmark.presentbench.loader import PresentBenchLoader


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_benchmark_loader(benchmark_id: str) -> BenchmarkLoader:
    if benchmark_id == "deck_bench":
        return DeckBenchLoader()
    if benchmark_id == "presentbench":
        return PresentBenchLoader()
    raise ValueError(f"Unsupported benchmark `{benchmark_id}`")


def init_eval_storage(config: EvalConfig) -> EvalDatabase:
    ensure_eval_directories(config)
    return EvalDatabase(config.paths.eval_db)


def write_json_artifact(output_path: str | Path, payload: Any) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(payload):
        serializable = asdict(payload)
    else:
        serializable = payload
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return str(path)


def clone_research_db(source_path: str | Path, output_dir: str | Path) -> Path:
    source = Path(source_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Research DB not found: {source}")
    destination = Path(output_dir).expanduser().resolve() / f"{uuid.uuid4()}.sqlite"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination

