from __future__ import annotations

import json
import base64
import io
import re
import shutil
import subprocess
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable

import yaml

from eval.metrics.benchmark.presentbench.metrics import (
    content_completeness_metric,
    content_correctness_metric,
    content_fidelity_metric,
    presentation_fundamentals_metric,
    presentbench_overall_metric,
    visual_design_and_layout_metric,
)
from eval.metrics.benchmark.presentbench.prompts import (
    PRESENTBENCH_JUDGE_SYSTEM_PROMPT,
    build_presentbench_user_prompt,
)
from eval.schema import MetricResult, Task, Transcript


JudgeRunner = Callable[[str, dict[str, Any]], tuple[str, str]]


@dataclass(slots=True)
class PresentBenchMetricSuite:
    llm_config_path: str | None
    secondary_output_dir: Path
    judge_runner: JudgeRunner | None = None

    def grade_transcript(
        self,
        transcript: Transcript,
        task: Task,
        candidate_deck_path: Path,
        source_text: str,
    ) -> list[MetricResult]:
        candidate_context = self._prepare_candidate_context(transcript, candidate_deck_path)
        verdicts = self._judge_checklists(task, source_text, candidate_context)
        weights = self._load_weights(task)
        created_at = datetime.now(timezone.utc).isoformat()
        artifact_path = self._write_verdict_artifact(
            transcript,
            verdicts,
            weights,
            candidate_deck_path,
            candidate_context,
        )

        results = [
            self._make_result(
                transcript,
                "presentation_fundamentals",
                presentation_fundamentals_metric(verdicts, weights),
                created_at,
                metadata={"verdict_artifact_path": artifact_path},
            ),
            self._make_result(
                transcript,
                "visual_design_and_layout",
                visual_design_and_layout_metric(verdicts, weights),
                created_at,
                metadata={"verdict_artifact_path": artifact_path},
            ),
            self._make_result(
                transcript,
                "content_completeness",
                content_completeness_metric(verdicts, weights),
                created_at,
                metadata={"verdict_artifact_path": artifact_path},
            ),
            self._make_result(
                transcript,
                "content_correctness",
                content_correctness_metric(verdicts, weights),
                created_at,
                metadata={"verdict_artifact_path": artifact_path},
            ),
            self._make_result(
                transcript,
                "content_fidelity",
                content_fidelity_metric(verdicts, weights),
                created_at,
                metadata={"verdict_artifact_path": artifact_path},
            ),
            self._make_result(
                transcript,
                "presentbench_overall",
                presentbench_overall_metric(verdicts, weights),
                created_at,
                artifact_path=artifact_path,
                metadata={"verdict_artifact_path": artifact_path},
            ),
        ]
        return results

    def _judge_checklists(self, task: Task, source_text: str, candidate_context: dict[str, Any]) -> dict[str, Any]:
        material_independent, material_dependent = self._load_checklists(task, Path(candidate_context["candidate_deck_path"]))
        runner = self.judge_runner or self._run_llm_judge

        verdicts = {
            "material_independent": self._evaluate_section(
                material_independent,
                build_payload=lambda: {
                    "candidate_deck_path": candidate_context["candidate_deck_path"],
                    "candidate_content_items": candidate_context["candidate_content_items"],
                    "candidate_pdf_path": candidate_context["candidate_pdf_path"],
                },
                judge_runner=runner,
            ),
            "material_dependent": self._evaluate_section(
                material_dependent,
                build_payload=lambda: {
                    "candidate_deck_path": candidate_context["candidate_deck_path"],
                    "candidate_content_items": candidate_context["candidate_content_items"],
                    "candidate_pdf_path": candidate_context["candidate_pdf_path"],
                    "source_text": source_text,
                    "material_path": task.metadata.get("material_path"),
                },
                judge_runner=runner,
            ),
        }
        return verdicts

    def _load_checklists(self, task: Task, candidate_deck_path: Path) -> tuple[list[list[Any]], list[list[Any]]]:
        judge_prompt = json.loads(Path(task.metadata["judge_prompt_path"]).read_text(encoding="utf-8"))
        common_prompt = json.loads(Path(task.metadata["common_judge_prompt_path"]).read_text(encoding="utf-8"))

        def merged(key: str) -> Any:
            if key in judge_prompt:
                return judge_prompt[key]
            return common_prompt[key]

        material_independent_prefix = self._decode_prompt_value(merged("material_independent_prefix"))
        material_dependent_prefix = self._decode_prompt_value(merged("material_dependent_prefix"))
        material_independent = [
            self._prefix_checklist(self._decode_prompt_value(merged("material_independent_checklist_1")), material_independent_prefix),
            self._prefix_checklist(self._decode_prompt_value(merged("material_independent_checklist_2")), material_independent_prefix),
        ]
        material_dependent = [
            self._prefix_checklist(self._decode_prompt_value(merged("material_dependent_checklist_1")), material_dependent_prefix),
            self._prefix_checklist(self._decode_prompt_value(merged("material_dependent_checklist_2")), material_dependent_prefix),
            self._prefix_checklist(self._decode_prompt_value(merged("material_dependent_checklist_3")), material_dependent_prefix),
        ]
        material_dependent[2] = material_dependent[2][: self._count_slides(candidate_deck_path)]
        return material_independent, material_dependent

    def _load_weights(self, task: Task) -> dict[str, Any]:
        return yaml.safe_load(Path(task.metadata["weights_path"]).read_text(encoding="utf-8")) or {}

    def _evaluate_section(
        self,
        checklists: list[list[Any]],
        build_payload: Callable[[], dict[str, Any]],
        judge_runner: JudgeRunner,
    ) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for outer_index, checklist in enumerate(checklists, start=1):
            class_results: dict[str, Any] = {}
            for inner_index, item in enumerate(checklist, start=1):
                item_key = f"{outer_index}.{inner_index}"
                if callable(item):
                    class_results[item_key] = item(build_payload()["candidate_deck_path"])
                    continue
                payload = build_payload()
                answer, explanation = judge_runner(item, payload)
                class_results[item_key] = {
                    "answer": answer.lower(),
                    "explanation": explanation,
                }
            results[str(outer_index)] = class_results
        return results

    def _run_llm_judge(self, prompt_text: str, payload: dict[str, Any]) -> tuple[str, str]:
        if self.llm_config_path is None:
            raise ValueError("PresentBenchMetricSuite requires llm_config_path when judge_runner is not provided")
        from src.llm.llm import LLMConfig, get_llm, init_from_config

        init_from_config(self.llm_config_path)
        llm = get_llm(LLMConfig(model="evaluator", temperature=0.0))
        content = [
            {
                "type": "text",
                "text": build_presentbench_user_prompt(prompt_text, payload.get("source_text")),
            },
            *payload["candidate_content_items"],
        ]
        response = llm.complete(
            [
                {"role": "system", "content": PRESENTBENCH_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=300,
        )
        match = re.search(r"\\boxed\{\s*(YES|NO)\s*\}", response, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().lower(), response
        raise ValueError(f"Judge response missing boxed verdict: {response}")

    def _write_verdict_artifact(
        self,
        transcript: Transcript,
        verdicts: dict[str, Any],
        weights: dict[str, Any],
        candidate_deck_path: Path,
        candidate_context: dict[str, Any],
    ) -> str:
        self.secondary_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.secondary_output_dir / f"{transcript.transcript_id}_presentbench_verdicts.json"
        payload = {
            **verdicts,
            "weights": weights,
            "candidate_deck_path": str(candidate_deck_path),
            "candidate_pdf_path": candidate_context["candidate_pdf_path"],
            "transcript_id": transcript.transcript_id,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(output_path)

    def _prepare_candidate_context(self, transcript: Transcript, candidate_deck_path: Path) -> dict[str, Any]:
        if self.judge_runner is not None:
            return {
                "candidate_deck_path": str(candidate_deck_path),
                "candidate_pdf_path": None,
                "candidate_content_items": [],
            }
        transcript_dir = self.secondary_output_dir / transcript.transcript_id
        transcript_dir.mkdir(parents=True, exist_ok=True)
        candidate_pdf_path = self._resolve_candidate_pdf(candidate_deck_path, transcript_dir)
        candidate_content_items = self._render_pdf_to_image_content(candidate_pdf_path)
        return {
            "candidate_deck_path": str(candidate_deck_path),
            "candidate_pdf_path": str(candidate_pdf_path),
            "candidate_content_items": candidate_content_items,
        }

    @staticmethod
    def _prefix_checklist(items: list[Any], prefix: str) -> list[Any]:
        return [prefix + item if isinstance(item, str) else item for item in items]

    @staticmethod
    def _decode_prompt_value(value: Any) -> Any:
        if isinstance(value, list):
            return [PresentBenchMetricSuite._decode_prompt_value(item) for item in value]
        if isinstance(value, dict):
            kind = value.get("__type__")
            if kind == "partial":
                function_ref = str(value.get("func") or "")
                args = [PresentBenchMetricSuite._decode_prompt_value(item) for item in value.get("args") or []]
                keywords = {
                    key: PresentBenchMetricSuite._decode_prompt_value(item)
                    for key, item in (value.get("keywords") or {}).items()
                }
                return partial(PresentBenchMetricSuite._import_compat_callable(function_ref), *args, **keywords)
            if kind == "callable":
                return PresentBenchMetricSuite._import_compat_callable(str(value.get("callable") or ""))
            return {
                key: PresentBenchMetricSuite._decode_prompt_value(item)
                for key, item in value.items()
            }
        return value

    @staticmethod
    def _import_compat_callable(qualname: str) -> Callable[..., dict[str, Any]]:
        if qualname == "utils.count_pages.check_slide_count":
            return PresentBenchMetricSuite._check_slide_count
        raise ValueError(f"Unsupported PresentBench callable reference: {qualname}")

    @staticmethod
    def _check_slide_count(slides_path: str, min_count: int, max_count: int) -> dict[str, Any]:
        num_slides = PresentBenchMetricSuite._count_slides(Path(slides_path))
        if min_count <= num_slides <= max_count:
            answer = "yes"
            explanation = (
                f"[judged by code] The number of slides is {num_slides}, which is within the required range "
                f"of {min_count}-{max_count}."
            )
        elif num_slides < min_count:
            answer = "no"
            explanation = (
                f"[judged by code] The number of slides is {num_slides}, which is too few. "
                f"The required range is {min_count}-{max_count} slides."
            )
        else:
            answer = "no"
            explanation = (
                f"[judged by code] The number of slides is {num_slides}, which is too many. "
                f"The required range is {min_count}-{max_count} slides."
            )
        return {"answer": answer, "explanation": explanation}

    @staticmethod
    def _count_slides(candidate_deck_path: Path) -> int:
        suffix = candidate_deck_path.suffix.lower()
        if suffix == ".pptx":
            with zipfile.ZipFile(candidate_deck_path, "r") as archive:
                return len(
                    [
                        name
                        for name in archive.namelist()
                        if name.startswith("ppt/slides/slide") and name.endswith(".xml")
                    ]
                )
        if suffix == ".pdf":
            try:
                import pypdfium2 as pdfium
            except ImportError as exc:
                raise ImportError("pypdfium2 is required to count PresentBench PDF pages") from exc
            with pdfium.PdfDocument(str(candidate_deck_path)) as pdf:
                return len(pdf)
        raise ValueError(f"Unsupported candidate deck format for PresentBench: {candidate_deck_path}")

    @staticmethod
    def _resolve_candidate_pdf(candidate_deck_path: Path, output_dir: Path) -> Path:
        candidate_deck_path = candidate_deck_path.expanduser().resolve()
        if candidate_deck_path.suffix.lower() == ".pdf":
            return candidate_deck_path

        sibling_pdf = candidate_deck_path.with_suffix(".pdf")
        if sibling_pdf.exists():
            return sibling_pdf

        office_binary = shutil.which("libreoffice") or shutil.which("soffice")
        if office_binary is None:
            raise FileNotFoundError(
                "PresentBench grading requires a rendered deck PDF. "
                f"No sibling PDF found for {candidate_deck_path.name}, and LibreOffice is not installed "
                "to convert the generated PPTX."
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                office_binary,
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(output_dir),
                str(candidate_deck_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        converted_pdf = output_dir / f"{candidate_deck_path.stem}.pdf"
        if not converted_pdf.exists():
            raise FileNotFoundError(f"LibreOffice did not produce the expected PDF: {converted_pdf}")
        return converted_pdf

    @staticmethod
    def _render_pdf_to_image_content(pdf_path: Path) -> list[dict[str, Any]]:
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise ImportError("pypdfium2 is required to render PresentBench deck PDFs for judging") from exc

        content_items: list[dict[str, Any]] = []
        with pdfium.PdfDocument(str(pdf_path)) as pdf:
            for page_index in range(len(pdf)):
                bitmap = pdf[page_index].render(scale=2)
                image = bitmap.to_pil()
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
                content_items.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded}",
                        },
                    }
                )
        return content_items

    @staticmethod
    def _make_result(
        transcript: Transcript,
        metric_id: str,
        scalar_value: float,
        created_at: str,
        artifact_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MetricResult:
        del created_at
        return MetricResult(
            metric_result_id=str(uuid.uuid4()),
            transcript_id=transcript.transcript_id,
            suite_id=transcript.suite_id,
            trial_index=transcript.trial_index,
            variant_id=transcript.variant_id,
            graph_version=transcript.graph_version,
            doc_pipeline_version=transcript.doc_pipeline_version,
            benchmark_id=transcript.benchmark_id,
            metric_id=metric_id,
            grader_id="presentbench_v1",
            subject_type="generated_deck",
            subject_id=transcript.transcript_id,
            status="success",
            scalar_value=scalar_value,
            reason=None,
            artifact_path=artifact_path,
            metadata=metadata or {},
        )
