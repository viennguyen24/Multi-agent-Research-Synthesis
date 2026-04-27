from __future__ import annotations

import json
import shutil
from pathlib import Path

from eval.reports.base import BenchmarkLoadResult, BenchmarkLoader
from eval.schema import Task


class PresentBenchLoader(BenchmarkLoader):
    benchmark_id = "presentbench"

    def load(self, manifest_path: str | Path) -> BenchmarkLoadResult:
        manifest_file = Path(manifest_path).expanduser().resolve()
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))

        dataset_root = self._resolve_dataset_root(manifest_file, str(payload["dataset_root"]))
        case_paths = payload.get("cases") or []
        if not case_paths:
            return BenchmarkLoadResult(benchmark_id=self.benchmark_id, tasks=[])

        common_judge_prompt_path = dataset_root / "common_judge_prompt.json"
        weights_path = dataset_root / "judge_weights.yaml"
        alias_dir = manifest_file.parent / "_material_aliases"
        alias_dir.mkdir(parents=True, exist_ok=True)

        tasks: list[Task] = []
        for relative_case in case_paths:
            case_root = dataset_root / str(relative_case)
            conference = case_root.parent.name.strip()
            paper_id = case_root.name.strip()
            material_path = case_root / "material.pdf"
            instructions_path = case_root / "generation_task" / "instructions.md"
            judge_prompt_path = case_root / "generation_task" / "judge_prompt.json"

            alias_name = f"{conference.lower()}__{paper_id}.pdf"
            aliased_material_path = alias_dir / alias_name
            self._ensure_material_alias(material_path, aliased_material_path)

            tasks.append(
                Task(
                    benchmark_id=self.benchmark_id,
                    suite_id=conference.lower(),
                    task_id=paper_id,
                    query=instructions_path.read_text(encoding="utf-8").strip(),
                    source_document_id=aliased_material_path.stem,
                    source_document_paths=[str(aliased_material_path)],
                    key_facts=[],
                    raw_reference_deck_path=None,
                    metadata={
                        "case_root": str(case_root),
                        "case_relative_path": str(relative_case),
                        "conference": conference,
                        "paper_id": paper_id,
                        "material_path": str(material_path),
                        "instructions_path": str(instructions_path),
                        "judge_prompt_path": str(judge_prompt_path),
                        "common_judge_prompt_path": str(common_judge_prompt_path),
                        "weights_path": str(weights_path),
                    },
                )
            )
        return BenchmarkLoadResult(benchmark_id=self.benchmark_id, tasks=tasks)

    @staticmethod
    def _ensure_material_alias(source_path: Path, alias_path: Path) -> None:
        if not source_path.exists():
            raise FileNotFoundError(f"PresentBench material not found: {source_path}")
        if alias_path.exists():
            source_stat = source_path.stat()
            alias_stat = alias_path.stat()
            if alias_stat.st_mtime >= source_stat.st_mtime and alias_stat.st_size == source_stat.st_size:
                return
        shutil.copy2(source_path, alias_path)

    @staticmethod
    def _resolve_dataset_root(manifest_file: Path, raw_dataset_root: str) -> Path:
        dataset_root = Path(raw_dataset_root).expanduser()
        if dataset_root.is_absolute():
            return dataset_root.resolve()

        candidate_paths = [(manifest_file.parent / dataset_root).resolve()]
        candidate_paths.extend((parent / dataset_root).resolve() for parent in manifest_file.parents)
        for candidate in candidate_paths:
            if candidate.exists():
                return candidate
        return candidate_paths[0]
