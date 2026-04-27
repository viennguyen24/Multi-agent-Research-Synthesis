from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys

from pypdf import PdfWriter
from pydantic import BaseModel

from eval.config import load_eval_config
from eval.db import EvalDatabase
from eval.metrics.benchmark.presentbench.metric_suite import PresentBenchMetricSuite
from eval.pipeline.common import write_json_artifact
from eval.pipeline.deck_views import build_deck_views
from eval.pipeline.documents import (
    _default_document_processor,
    build_documents,
    download_documents,
    process_documents,
    register_benchmark_tasks,
)
from eval.pipeline.grader import run_grader
from eval.pipeline.harness import run_harness
from eval.pipeline.reports import run_reports
from eval.schema import MetricResult, Task, Transcript


class TestEvalPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmpdir.name)
        (self.root / "eval/metrics/benchmark/deck_bench/data").mkdir(parents=True, exist_ok=True)
        (self.root / "data").mkdir(parents=True, exist_ok=True)
        (self.root / "output").mkdir(parents=True, exist_ok=True)
        self.research_db_path = self.root / "data/research.db"
        sqlite3.connect(self.research_db_path).close()

        manifest = {
            "tasks": {
                "1041": {
                    "conference": "ECCV",
                    "year": "2024",
                    "paper_url": str(self.root / "paper.pdf"),
                    "slides_url": str(self.root / "reference.pdf"),
                }
            }
        }
        (self.root / "eval/metrics/benchmark/deck_bench/data/tasks.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        (self.root / "paper.pdf").write_bytes(b"%PDF-1.4\n% test paper\n")
        self._write_config()
        self._write_fake_pdf(self.root / "reference.pdf")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_register_and_build_documents_stage_isolated(self) -> None:
        config = load_eval_config(self.root / "config.yaml")
        captured: list[list[str]] = []

        def fake_processor(pdf_paths: list[str], _llm_config_path: str | None) -> object:
            captured.append(pdf_paths)
            return SimpleNamespace(
                processed_documents=[
                    {
                        "doc_id": "doc-1",
                        "source_path": str(self.root / "paper.pdf"),
                        "paper_title": "paper",
                    }
                ]
            )

        processed_paths = build_documents(
            config=config,
            benchmark_id="deck_bench",
            processor=fake_processor,
        )
        self.assertEqual(processed_paths, [str(self.root / "paper.pdf")])
        self.assertEqual(captured, [[str(self.root / "paper.pdf")]])

        with EvalDatabase(config.paths.eval_db) as db:
            tasks = db.list_tasks("deck_bench")
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].task_id, "1041")
            self.assertEqual(tasks[0].suite_id, "eccv_2024")
            self.assertEqual(tasks[0].query, "Explain this paper to an audience of laypeople")
            self.assertEqual(tasks[0].source_document_id, "doc-1")
            self.assertEqual(tasks[0].source_document_paths, [str(self.root / "paper.pdf")])
            self.assertEqual(tasks[0].raw_reference_deck_path, str(self.root / "reference.pdf"))

    def test_download_documents_populates_local_paths_without_processing(self) -> None:
        config = load_eval_config(self.root / "config.yaml")

        downloaded_paths = download_documents(config=config, benchmark_id="deck_bench")
        self.assertEqual(downloaded_paths, [str(self.root / "paper.pdf")])

        with EvalDatabase(config.paths.eval_db) as db:
            tasks = db.list_tasks("deck_bench")
            self.assertEqual(tasks[0].source_document_id, "1041")
            self.assertEqual(tasks[0].source_document_paths, [str(self.root / "paper.pdf")])
            self.assertEqual(tasks[0].raw_reference_deck_path, str(self.root / "reference.pdf"))

    def test_process_documents_uses_downloaded_local_paths(self) -> None:
        config = load_eval_config(self.root / "config.yaml")
        download_documents(config=config, benchmark_id="deck_bench")
        captured: list[list[str]] = []

        def fake_processor(pdf_paths: list[str], _llm_config_path: str | None) -> object:
            captured.append(pdf_paths)
            self._seed_processed_document("doc-1", self.root / "paper.pdf")
            return SimpleNamespace(
                processed_documents=[
                    {
                        "doc_id": "doc-1",
                        "source_path": str(self.root / "paper.pdf"),
                        "paper_title": "paper",
                    }
                ]
            )

        processed_paths = process_documents(config=config, benchmark_id="deck_bench", processor=fake_processor)
        self.assertEqual(processed_paths, [str(self.root / "paper.pdf")])
        self.assertEqual(captured, [[str(self.root / "paper.pdf")]])

        with EvalDatabase(config.paths.eval_db) as db:
            tasks = db.list_tasks("deck_bench")
            self.assertEqual(tasks[0].source_document_id, "doc-1")
            self.assertEqual(tasks[0].source_document_paths, [str(self.root / "paper.pdf")])

    def test_process_documents_recovers_persisted_doc_when_runner_returns_none(self) -> None:
        config = load_eval_config(self.root / "config.yaml")
        download_documents(config=config, benchmark_id="deck_bench")

        def fake_processor(pdf_paths: list[str], _llm_config_path: str | None) -> object:
            self._seed_processed_document("doc-1", self.root / "paper.pdf")
            return SimpleNamespace(processed_documents=[])

        processed_paths = process_documents(config=config, benchmark_id="deck_bench", processor=fake_processor)
        self.assertEqual(processed_paths, [str(self.root / "paper.pdf")])

        with EvalDatabase(config.paths.eval_db) as db:
            tasks = db.list_tasks("deck_bench")
            self.assertEqual(tasks[0].source_document_id, "doc-1")

    def test_build_documents_skips_existing_processed_document_by_default(self) -> None:
        config = load_eval_config(self.root / "config.yaml")

        def fake_processor(pdf_paths: list[str], _llm_config_path: str | None) -> object:
            self._seed_processed_document("doc-1", self.root / "paper.pdf")
            return SimpleNamespace(
                processed_documents=[
                    {
                        "doc_id": "doc-1",
                        "source_path": str(self.root / "paper.pdf"),
                        "paper_title": "paper",
                    }
                ]
            )

        first_paths = build_documents(config=config, benchmark_id="deck_bench", processor=fake_processor)
        self.assertEqual(first_paths, [str(self.root / "paper.pdf")])

        captured: list[list[str]] = []

        def should_not_run(pdf_paths: list[str], _llm_config_path: str | None) -> object:
            captured.append(pdf_paths)
            return SimpleNamespace(processed_documents=[])

        second_paths = build_documents(config=config, benchmark_id="deck_bench", processor=should_not_run)
        self.assertEqual(second_paths, [])
        self.assertEqual(captured, [])

        with EvalDatabase(config.paths.eval_db) as db:
            tasks = db.list_tasks("deck_bench")
            self.assertEqual(tasks[0].source_document_id, "doc-1")
            self.assertEqual(tasks[0].source_document_paths, [str(self.root / "paper.pdf")])

    def test_build_documents_force_process_rebuilds_existing_document(self) -> None:
        config = load_eval_config(self.root / "config.yaml")

        def first_processor(pdf_paths: list[str], _llm_config_path: str | None) -> object:
            self._seed_processed_document("doc-1", self.root / "paper.pdf")
            return SimpleNamespace(
                processed_documents=[
                    {
                        "doc_id": "doc-1",
                        "source_path": str(self.root / "paper.pdf"),
                        "paper_title": "paper",
                    }
                ]
            )

        build_documents(config=config, benchmark_id="deck_bench", processor=first_processor)

        captured: list[list[str]] = []

        def second_processor(pdf_paths: list[str], _llm_config_path: str | None) -> object:
            captured.append(pdf_paths)
            self._seed_processed_document("doc-2", self.root / "paper.pdf")
            return SimpleNamespace(
                processed_documents=[
                    {
                        "doc_id": "doc-2",
                        "source_path": str(self.root / "paper.pdf"),
                        "paper_title": "paper-v2",
                    }
                ]
            )

        forced_paths = build_documents(
            config=config,
            benchmark_id="deck_bench",
            processor=second_processor,
            force_process=True,
        )
        self.assertEqual(forced_paths, [str(self.root / "paper.pdf")])
        self.assertEqual(captured, [[str(self.root / "paper.pdf")]])

        with EvalDatabase(config.paths.eval_db) as db:
            tasks = db.list_tasks("deck_bench")
            self.assertEqual(tasks[0].source_document_id, "doc-2")
            self.assertEqual(tasks[0].metadata["paper_title"], "paper-v2")

    def test_default_document_processor_keeps_contextualization_enabled(self) -> None:
        mocked = Mock(return_value=SimpleNamespace(processed_documents=[]))
        fake_module = SimpleNamespace(process_documents=mocked)
        with patch.dict(sys.modules, {"eval.graph_runner": fake_module}):
            _default_document_processor(["paper.pdf"], "eval/llm.config.yaml")

        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(
            mocked.call_args.kwargs,
            {
                "pdf_paths": ["paper.pdf"],
                "llm_config_path": "eval/llm.config.yaml",
                "database_path": None,
            },
        )

    def test_register_presentbench_tasks_uses_thin_manifest_and_unique_material_aliases(self) -> None:
        (self.root / "eval/metrics/benchmark/presentbench/data/academia/ICLR_2025/paper_one/generation_task").mkdir(
            parents=True,
            exist_ok=True,
        )
        (self.root / "eval/metrics/benchmark/presentbench/data/academia/NeurIPS_2024/paper_two/generation_task").mkdir(
            parents=True,
            exist_ok=True,
        )

        common_prompt = {
            "material_independent_prefix": "MI: ",
            "material_dependent_prefix": "MD: ",
            "material_independent_checklist_1": ["Question A"],
            "material_independent_checklist_2": ["Question B"],
            "material_dependent_checklist_1": ["Question C"],
            "material_dependent_checklist_2": ["Question D"],
            "material_dependent_checklist_3": ["Question E"],
        }
        weights = {
            "material_independent": {"1": 20.0, "2": 20.0},
            "material_dependent": {"1": 20.0, "2": 20.0, "3": 20.0},
        }
        manifest = {
            "dataset_root": "eval/metrics/benchmark/presentbench/data/academia",
            "cases": [
                "ICLR_2025/paper_one",
                "NeurIPS_2024/paper_two",
            ],
        }

        dataset_root = self.root / "eval/metrics/benchmark/presentbench/data/academia"
        (dataset_root / "common_judge_prompt.json").write_text(json.dumps(common_prompt), encoding="utf-8")
        (dataset_root / "judge_weights.yaml").write_text(
            "material_independent:\n  '1': 20.0\n  '2': 20.0\nmaterial_dependent:\n  '1': 20.0\n  '2': 20.0\n  '3': 20.0\n",
            encoding="utf-8",
        )
        (self.root / "eval/metrics/benchmark/presentbench/data/tasks.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        del weights

        for relative_case in manifest["cases"]:
            case_root = dataset_root / relative_case
            (case_root / "material.pdf").write_bytes(b"%PDF-1.4\n%stub\n")
            (case_root / "generation_task/instructions.md").write_text(
                f"Create a technical deck for {case_root.name}.",
                encoding="utf-8",
            )
            (case_root / "generation_task/judge_prompt.json").write_text(
                json.dumps({"material_dependent_checklist_3": ["Per-slide fidelity question"]}),
                encoding="utf-8",
            )

        config = load_eval_config(self.root / "config.yaml")
        tasks = register_benchmark_tasks(config, "presentbench")

        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0].benchmark_id, "presentbench")
        self.assertEqual(tasks[0].suite_id, "iclr_2025")
        self.assertEqual(tasks[0].task_id, "paper_one")
        self.assertEqual(tasks[0].query, "Create a technical deck for paper_one.")
        self.assertEqual(tasks[1].suite_id, "neurips_2024")
        self.assertNotEqual(tasks[0].source_document_id, tasks[1].source_document_id)
        self.assertTrue(tasks[0].source_document_paths[0].endswith(".pdf"))
        self.assertTrue(Path(tasks[0].source_document_paths[0]).exists())
        self.assertEqual(
            tasks[0].metadata["judge_prompt_path"],
            str(dataset_root / "ICLR_2025/paper_one/generation_task/judge_prompt.json"),
        )

    def test_harness_persists_append_only_transcripts(self) -> None:
        config = load_eval_config(self.root / "config.yaml")
        def fake_processor(pdf_paths: list[str], _llm_config_path: str | None) -> object:
            return SimpleNamespace(
                processed_documents=[
                    {
                        "doc_id": "doc-1",
                        "source_path": str(self.root / "paper.pdf"),
                        "paper_title": "paper",
                    }
                ]
            )

        build_documents(config, "deck_bench", processor=fake_processor)
        generated_pptx = self.root / "output/generated.pptx"
        self._write_fake_pptx(generated_pptx, [["Gen title", "Gen bullet"]])

        runner_calls: list[dict[str, object]] = []

        def fake_runner(**kwargs):
            runner_calls.append(kwargs)
            return SimpleNamespace(
                session_id="sess-1",
                status="success",
                final_state={"review": {"export_ready": True}, "messages": []},
                node_events=[],
                pptx_path=str(generated_pptx),
                final_warnings=[],
                error_text=None,
            )

        transcripts = run_harness(
            config=config,
            benchmark_id="deck_bench",
            variant_id="v1",
            graph_version="g1",
            doc_pipeline_version="d1",
            trial_count=2,
            runner=fake_runner,
        )
        self.assertEqual(len(transcripts), 2)
        self.assertEqual(runner_calls[0]["doc_ids"], ["doc-1"])
        with EvalDatabase(config.paths.eval_db) as db:
            stored = db.list_transcripts(benchmark_id="deck_bench")
            self.assertEqual(len(stored), 2)

    def test_harness_accepts_plain_dict_node_events(self) -> None:
        config = load_eval_config(self.root / "config.yaml")
        register_benchmark_tasks(config, "deck_bench")
        generated_pptx = self.root / "output/generated.pptx"
        self._write_fake_pptx(generated_pptx, [["Gen title", "Gen bullet"]])

        def fake_runner(**_kwargs):
            return SimpleNamespace(
                session_id="sess-1",
                status="success",
                final_state={"review": {"export_ready": True}, "messages": []},
                node_events=[{"node_name": "planner", "event_type": "end"}],
                pptx_path=str(generated_pptx),
                final_warnings=[],
                error_text=None,
            )

        transcripts = run_harness(
            config=config,
            benchmark_id="deck_bench",
            variant_id="v1",
            graph_version="g1",
            doc_pipeline_version="d1",
            trial_count=1,
            runner=fake_runner,
        )

        self.assertEqual(transcripts[0].status, "success")
        self.assertIsNotNone(transcripts[0].node_events_artifact_path)
        node_events_path = Path(transcripts[0].node_events_artifact_path or "")
        self.assertTrue(node_events_path.exists())
        payload = json.loads(node_events_path.read_text(encoding="utf-8"))
        self.assertEqual(payload, [{"node_name": "planner", "event_type": "end"}])

    def test_build_deck_views_extracts_generated_and_reference_views(self) -> None:
        config = load_eval_config(self.root / "config.yaml")
        def fake_processor(pdf_paths: list[str], _llm_config_path: str | None) -> object:
            return SimpleNamespace(
                processed_documents=[
                    {
                        "doc_id": "doc-1",
                        "source_path": str(self.root / "paper.pdf"),
                        "paper_title": "paper",
                    }
                ]
            )

        build_documents(config, "deck_bench", processor=fake_processor)
        generated_pptx = self.root / "output/generated.pptx"
        self._write_fake_pptx(generated_pptx, [["Gen title", "Gen bullet"]])

        with EvalDatabase(config.paths.eval_db) as db:
            db.insert_transcript(
                transcript=SimpleNamespace(
                    transcript_id="tx-1",
                    benchmark_id="deck_bench",
                    suite_id="eccv_2024",
                    task_id="1041",
                    trial_index=0,
                    variant_id="v1",
                    graph_version="g1",
                    doc_pipeline_version="d1",
                    status="success",
                    created_at="now",
                    finished_at="later",
                    session_id="sess",
                    query="Explain this paper to an audience of laypeople",
                    source_document_id="doc-1",
                    final_deck_path=str(generated_pptx),
                    transcript_artifact_path=None,
                    final_state_artifact_path=None,
                    node_events_artifact_path=None,
                    debug_artifact_path=None,
                    error_text=None,
                ),
                created_at="now",
            )

        outputs = build_deck_views(config, "deck_bench")
        self.assertEqual(len(outputs), 2)
        with EvalDatabase(config.paths.eval_db) as db:
            stored = db.list_deck_views()
            self.assertEqual(len(stored), 2)

    def test_reports_aggregate_metric_rows(self) -> None:
        config = load_eval_config(self.root / "config.yaml")
        with EvalDatabase(config.paths.eval_db) as db:
            db.insert_metric_result(
                result=SimpleNamespace(
                    metric_result_id="mr-1",
                    transcript_id="tx-1",
                    suite_id="eccv_2024",
                    trial_index=0,
                    variant_id="v1",
                    graph_version="g1",
                    doc_pipeline_version="d1",
                    benchmark_id="deck_bench",
                    metric_id="deck_fidelity",
                    grader_id="grader",
                    subject_type="generated_deck",
                    subject_id="tx-1",
                    status="success",
                    scalar_value=0.5,
                    pass_fail=None,
                    reason=None,
                    artifact_path=None,
                    metadata={},
                ),
                created_at="now",
            )
            db.insert_metric_result(
                result=SimpleNamespace(
                    metric_result_id="mr-2",
                    transcript_id="tx-2",
                    suite_id="eccv_2024",
                    trial_index=1,
                    variant_id="v1",
                    graph_version="g1",
                    doc_pipeline_version="d1",
                    benchmark_id="deck_bench",
                    metric_id="deck_fidelity",
                    grader_id="grader",
                    subject_type="generated_deck",
                    subject_id="tx-2",
                    status="success",
                    scalar_value=1.0,
                    pass_fail=None,
                    reason=None,
                    artifact_path=None,
                    metadata={},
                ),
                created_at="now",
            )

        report_path = run_reports(config, "deck_bench", suite_id="eccv_2024")
        content = Path(report_path).read_text(encoding="utf-8")
        self.assertIn("deck_fidelity", content)
        self.assertIn("0.75", content)

    def test_write_json_artifact_serializes_nested_pydantic_models(self) -> None:
        class ExampleModel(BaseModel):
            value: str

        artifact_path = write_json_artifact(
            self.root / "artifact.json",
            {"outer": ExampleModel(value="ok")},
        )
        payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
        self.assertEqual(payload, {"outer": {"value": "ok"}})

    def test_run_grader_presentbench_does_not_require_deck_views(self) -> None:
        config = load_eval_config(self.root / "config.yaml")
        task = Task(
            benchmark_id="presentbench",
            suite_id="iclr_2025",
            task_id="paper_one",
            query="Create a technical deck.",
            source_document_id="iclr_2025__paper_one",
            source_document_paths=[str(self.root / "paper_one.pdf")],
            metadata={
                "material_path": str(self.root / "paper_one.pdf"),
                "judge_prompt_path": str(self.root / "judge_prompt.json"),
                "common_judge_prompt_path": str(self.root / "common_judge_prompt.json"),
                "weights_path": str(self.root / "judge_weights.yaml"),
            },
        )
        transcript = Transcript(
            transcript_id="tx-presentbench",
            benchmark_id="presentbench",
            suite_id="iclr_2025",
            task_id="paper_one",
            trial_index=0,
            variant_id="baseline",
            graph_version="g1",
            doc_pipeline_version="d1",
            status="success",
            created_at="now",
            finished_at="later",
            session_id="sess-1",
            query=task.query,
            source_document_id=task.source_document_id,
            final_deck_path=str(self.root / "candidate.pptx"),
        )
        self._write_fake_pptx(Path(transcript.final_deck_path), [["Title", "Bullet"]])

        with EvalDatabase(config.paths.eval_db) as db:
            db.register_task(task, "now")
            db.insert_transcript(transcript, "now")

        fake_results = [
            MetricResult(
                metric_result_id="mr-overall",
                transcript_id=transcript.transcript_id,
                suite_id=transcript.suite_id,
                trial_index=transcript.trial_index,
                variant_id=transcript.variant_id,
                graph_version=transcript.graph_version,
                doc_pipeline_version=transcript.doc_pipeline_version,
                benchmark_id=transcript.benchmark_id,
                metric_id="presentbench_overall",
                grader_id="presentbench_v1",
                subject_type="generated_deck",
                subject_id=transcript.transcript_id,
                status="success",
                scalar_value=75.0,
                artifact_path=None,
                metadata={},
            )
        ]

        fake_research_db = SimpleNamespace(
            load_document=lambda _doc_id: SimpleNamespace(markdown="Paper background", source_chunks=[]),
            close=lambda: None,
        )

        with mock.patch(
            "eval.metrics.benchmark.presentbench.metric_suite.PresentBenchMetricSuite.grade_transcript",
            return_value=fake_results,
        ) as mocked_grade:
            written = run_grader(
                config,
                "presentbench",
                research_db_factory=lambda _path: fake_research_db,
            )

        self.assertEqual(written, 1)
        self.assertEqual(mocked_grade.call_count, 1)
        with EvalDatabase(config.paths.eval_db) as db:
            metric_rows = db.list_metric_results(benchmark_id="presentbench")
            self.assertEqual(len(metric_rows), 1)
            self.assertEqual(metric_rows[0]["metric_id"], "presentbench_overall")

    def test_presentbench_metric_suite_returns_six_aggregate_metrics_and_verdict_artifact(self) -> None:
        case_root = self.root / "academia/ICLR_2025/paper_one"
        (case_root / "generation_task").mkdir(parents=True, exist_ok=True)
        (case_root / "material.pdf").write_bytes(b"%PDF-1.4\n%stub\n")
        self._write_fake_pptx(self.root / "candidate.pptx", [["Title", "Bullet"]])

        common_prompt = {
            "material_independent_prefix": "MI: ",
            "material_dependent_prefix": "MD: ",
            "material_independent_checklist_1": ["Fundamentals question"],
            "material_independent_checklist_2": ["Visual question"],
            "material_dependent_checklist_1": ["Completeness question"],
            "material_dependent_checklist_2": ["Correctness question"],
            "material_dependent_checklist_3": ["Fidelity question 1", "Fidelity question 2"],
        }
        (case_root.parent.parent / "common_judge_prompt.json").write_text(
            json.dumps(common_prompt),
            encoding="utf-8",
        )
        (case_root.parent.parent / "judge_weights.yaml").write_text(
            "material_independent:\n  '1': 20.0\n  '2': 20.0\nmaterial_dependent:\n  '1': 20.0\n  '2': 20.0\n  '3': 20.0\n",
            encoding="utf-8",
        )
        (case_root / "generation_task/judge_prompt.json").write_text("{}", encoding="utf-8")

        task = Task(
            benchmark_id="presentbench",
            suite_id="iclr_2025",
            task_id="paper_one",
            query="Create a technical deck.",
            source_document_id="iclr_2025__paper_one",
            source_document_paths=[str(case_root / "material.pdf")],
            metadata={
                "material_path": str(case_root / "material.pdf"),
                "judge_prompt_path": str(case_root / "generation_task/judge_prompt.json"),
                "common_judge_prompt_path": str(case_root.parent.parent / "common_judge_prompt.json"),
                "weights_path": str(case_root.parent.parent / "judge_weights.yaml"),
            },
        )
        transcript = Transcript(
            transcript_id="tx-1",
            benchmark_id="presentbench",
            suite_id="iclr_2025",
            task_id="paper_one",
            trial_index=0,
            variant_id="baseline",
            graph_version="g1",
            doc_pipeline_version="d1",
            status="success",
            created_at="now",
            finished_at="later",
            session_id="sess-1",
            query=task.query,
            source_document_id=task.source_document_id,
            final_deck_path=str(self.root / "candidate.pptx"),
        )

        answers = {
            "MI: Fundamentals question": ("yes", "good progression"),
            "MI: Visual question": ("no", "layout is crowded"),
            "MD: Completeness question": ("yes", "main findings covered"),
            "MD: Correctness question": ("no", "one claim is unsupported"),
            "MD: Fidelity question 1": ("yes", "aligned to slide 1"),
        }

        suite = PresentBenchMetricSuite(
            llm_config_path=None,
            secondary_output_dir=self.root / "artifacts/metric_results",
            judge_runner=lambda prompt_text, _payload: answers[prompt_text],
        )
        results = suite.grade_transcript(
            transcript=transcript,
            task=task,
            candidate_deck_path=Path(transcript.final_deck_path),
            source_text="Paper background text",
        )

        metric_ids = {result.metric_id for result in results}
        self.assertEqual(
            metric_ids,
            {
                "presentation_fundamentals",
                "visual_design_and_layout",
                "content_completeness",
                "content_correctness",
                "content_fidelity",
                "presentbench_overall",
            },
        )
        score_by_metric = {result.metric_id: result.scalar_value for result in results}
        self.assertEqual(score_by_metric["presentation_fundamentals"], 100.0)
        self.assertEqual(score_by_metric["visual_design_and_layout"], 0.0)
        self.assertEqual(score_by_metric["content_completeness"], 100.0)
        self.assertEqual(score_by_metric["content_correctness"], 0.0)
        self.assertEqual(score_by_metric["content_fidelity"], 100.0)
        self.assertEqual(score_by_metric["presentbench_overall"], 60.0)

        overall = next(result for result in results if result.metric_id == "presentbench_overall")
        self.assertTrue(overall.artifact_path)
        verdict_dump = json.loads(Path(overall.artifact_path).read_text(encoding="utf-8"))
        self.assertIn("material_independent", verdict_dump)
        self.assertEqual(
            verdict_dump["material_dependent"]["3"]["3.1"]["answer"],
            "yes",
        )
        self.assertNotIn("3.2", verdict_dump["material_dependent"]["3"])

    def _write_config(self) -> None:
        config_text = f"""
llm_config: llm.config.yaml
storage:
  eval_db: data/eval.db
  artifacts_dir: artifacts
  artifacts:
    transcripts: artifacts/transcripts
    deck_views: artifacts/deck_views
    metric_results: artifacts/metric_results
    reports: artifacts/reports
    runtime_dbs: artifacts/runtime_dbs
runtime:
  research_db: data/research.db
  output_dir: output
benchmarks:
  deck_bench: eval/metrics/benchmark/deck_bench/data/tasks.json
  presentbench: eval/metrics/benchmark/presentbench/data/tasks.json
"""
        (self.root / "config.yaml").write_text(config_text.strip() + "\n", encoding="utf-8")
        (self.root / "llm.config.yaml").write_text("groups: {}\nproviders: {}\nlitellm: {}\n", encoding="utf-8")

    @staticmethod
    def _write_fake_pptx(path: Path, slides: list[list[str]]) -> None:
        with zipfile.ZipFile(path, "w") as archive:
            for index, texts in enumerate(slides, start=1):
                slide_xml = (
                    '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
                    'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
                    + "".join(f"<a:t>{text}</a:t>" for text in texts)
                    + "</p:sld>"
                )
                archive.writestr(f"ppt/slides/slide{index}.xml", slide_xml)

    @staticmethod
    def _write_fake_pdf(path: Path) -> None:
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        with path.open("wb") as handle:
            writer.write(handle)

    def _seed_processed_document(self, doc_id: str, source_path: Path) -> None:
        conn = sqlite3.connect(self.research_db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source_path TEXT,
                    filename TEXT,
                    markdown TEXT,
                    page_count INTEGER,
                    content_hash TEXT,
                    run_id TEXT,
                    created_at TEXT,
                    schema TEXT,
                    paper_metadata TEXT,
                    document_context TEXT
                )
                """
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (
                    id, source_path, filename, markdown, page_count, content_hash,
                    run_id, created_at, schema, paper_metadata, document_context
                ) VALUES (?, ?, ?, '', 1, '', '', '', '', '', '')
                """,
                (doc_id, str(source_path), source_path.name),
            )
            conn.commit()
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
