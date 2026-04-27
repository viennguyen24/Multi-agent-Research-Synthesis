from __future__ import annotations

import json
from pathlib import Path

from eval.reports.base import BenchmarkLoadResult, BenchmarkLoader
from eval.schema import Task


DEFAULT_DECKBENCH_QUERY = "Explain this paper to an audience of laypeople"


class DeckBenchLoader(BenchmarkLoader):
    benchmark_id = "deck_bench"

    def load(self, manifest_path: str | Path) -> BenchmarkLoadResult:
        path = Path(manifest_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        tasks: list[Task] = []
        raw_tasks = payload.get("tasks", payload)
        for task_id, item in raw_tasks.items():
            conference = str(item["conference"]).strip()
            year = str(item["year"]).strip()
            tasks.append(
                Task(
                    benchmark_id=self.benchmark_id,
                    suite_id=f"{conference.lower()}_{year}",
                    task_id=str(task_id),
                    query=DEFAULT_DECKBENCH_QUERY,
                    source_document_id=str(task_id),
                    source_document_paths=[],
                    source_paper_url=item["paper_url"],
                    key_facts=[],
                    raw_reference_deck_path=None,
                    metadata={
                        "paper_id": str(task_id),
                        "conference": conference,
                        "year": year,
                        "paper_url": item["paper_url"],
                        "slides_url": item["slides_url"],
                    },
                )
            )
        return BenchmarkLoadResult(benchmark_id=self.benchmark_id, tasks=tasks)
