from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eval.schema import Task


@dataclass(slots=True)
class BenchmarkLoadResult:
    benchmark_id: str
    tasks: list[Task]


class BenchmarkLoader:
    benchmark_id: str

    def load(self, manifest_path: str | Path) -> BenchmarkLoadResult:
        raise NotImplementedError

