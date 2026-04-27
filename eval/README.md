# Eval Pipeline

This repo has two separate execution surfaces:

- [`main.py`](/mnt/c/Users/nguye/OneDrive/Desktop/agentic_ai/eval/main.py): the normal CLI entrypoint that processes documents, runs the graph, and exports the final deck.
- [`eval/graph_runner.py`](/mnt/c/Users/nguye/OneDrive/Desktop/agentic_ai/eval/eval/graph_runner.py): the reusable in-process helper that lets the eval harness call that same graph path without shelling out through the CLI.

The eval subsystem exists to evaluate final rendered `.pptx` outputs while keeping eval storage, benchmark loading, grading, and reports isolated from the main app flow.

## Setup

1. Install dependencies from the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure runtime secrets in `.env` for the normal app path.

3. Configure eval LLM routing in [`eval/llm.config.yaml`](/mnt/c/Users/nguye/OneDrive/Desktop/agentic_ai/eval/eval/llm.config.yaml).

4. Register benchmark tasks in [`eval/metrics/benchmark/deck_bench/data/tasks.json`](/mnt/c/Users/nguye/OneDrive/Desktop/agentic_ai/eval/eval/metrics/benchmark/deck_bench/data/tasks.json).

## Commands

Initialize eval storage:

```bash
python scripts/eval/init_eval_db.py --config eval/config.yaml
```

Build processed documents for selected benchmark tasks:

```bash
python scripts/eval/build_documents.py --config eval/config.yaml --benchmark deck_bench
```

Run the harness against already-processed documents:

```bash
python scripts/eval/run_harness.py \
  --config eval/config.yaml \
  --benchmark deck_bench \
  --variant-id baseline \
  --graph-version graph_v1 \
  --doc-pipeline-version docs_v1 \
  --trial-count 1
```

Build metric-facing deck views:

```bash
python scripts/eval/build_deck_views.py --config eval/config.yaml --benchmark deck_bench
```

Run grading:

```bash
python scripts/eval/run_grader.py --config eval/config.yaml --benchmark deck_bench
```

Generate aggregate reports:

```bash
python scripts/eval/run_reports.py --config eval/config.yaml --benchmark deck_bench
```

## Tests

Run tests from the repo root.

In this checkout, the working interpreter is the project virtualenv Python:

```bash
./.venv/Scripts/python.exe -B -m unittest tests.test_eval_pipeline
```

Run one focused test:

```bash
./.venv/Scripts/python.exe -B -m unittest tests.test_eval_pipeline.TestEvalPipeline.test_register_and_build_documents_stage_isolated
```

Other useful focused slices:

```bash
./.venv/Scripts/python.exe -B -m unittest tests.test_eval_pipeline.TestEvalPipeline.test_harness_persists_append_only_transcripts
./.venv/Scripts/python.exe -B -m unittest tests.test_eval_pipeline.TestEvalPipeline.test_harness_accepts_plain_dict_node_events
./.venv/Scripts/python.exe -B -m unittest tests.test_eval_pipeline.TestEvalPipeline.test_build_deck_views_extracts_generated_and_reference_views
./.venv/Scripts/python.exe -B -m unittest tests.test_eval_pipeline.TestEvalPipeline.test_write_json_artifact_serializes_nested_pydantic_models
```

If you only want a syntax check over the eval code without running the pipeline tests:

```bash
./.venv/Scripts/python.exe -m py_compile eval/config.py eval/schema.py eval/graph_runner.py eval/pipeline/common.py eval/pipeline/documents.py eval/pipeline/harness.py eval/pipeline/deck_views.py eval/pipeline/grader.py eval/pipeline/reports.py eval/metrics/benchmark/deck_bench/loader.py
```

## DeckView Purpose

`DeckView` is the thin metric-facing projection of a deck artifact.

It exists because the raw `.pptx` file is the canonical grading target, but the metric layer should not need to parse presentation XML, know benchmark-specific file layouts, or depend on internal graph state. `DeckView` keeps only the information metrics actually need:

- slide order
- ordered text blocks per slide
- combined slide text
- source provenance for each block
- extraction version and source identity

That separation protects the metric API from future graph changes and keeps raw deck files as the ground truth.

## Dataflow

The stages are independent. Nothing auto-runs the next stage.

1. `build_documents`
   Input: benchmark task registrations with source document paths.
   Work: process raw source docs into the normal research DB.
   Output: processed documents and retrieval material in `data/research.db`.

2. `run_harness`
   Input: normalized tasks, manual `variant_id`, `graph_version`, `doc_pipeline_version`, and trial count.
   Work: clone the research DB per trial, run the real graph in-process through `eval/graph_runner.py`, observe streamed state, and persist immutable transcript artifacts.
   Output: transcript metadata in `data/eval.db`, transcript JSON artifacts in `artifacts/transcripts/`, generated `.pptx` decks in `output/`, and cloned trial DBs in `artifacts/runtime_dbs/`.

3. `build_deck_views`
   Input: generated deck outputs from transcripts and reference deck registrations from the benchmark loader.
   Work: extract ordered slide text blocks from each deck into a thin, immutable deck-view artifact.
   Output: deck-view metadata in `data/eval.db` and deck-view JSON artifacts in `artifacts/deck_views/`.

4. `run_grader`
   Input: stored transcripts, required deck views, and processed source documents.
   Work: run deterministic and LLM-judge metrics in read-only mode and persist one metric result per metric per subject.
   Output: metric-result metadata in `data/eval.db` and metric artifacts in `artifacts/metric_results/`.

5. `run_reports`
   Input: persisted metric results and transcript metadata.
   Work: aggregate results across variants, graph versions, document pipeline versions, suites, and trials.
   Output: report files in `artifacts/reports/`.

## Storage Layout

- [`eval/config.yaml`](/mnt/c/Users/nguye/OneDrive/Desktop/agentic_ai/eval/eval/config.yaml): eval storage and benchmark manifest wiring
- [`eval/llm.config.yaml`](/mnt/c/Users/nguye/OneDrive/Desktop/agentic_ai/eval/eval/llm.config.yaml): eval-only LiteLLM config for LLM-judge metrics
- `data/eval.db`: queryable metadata store for tasks, transcripts, deck views, metric results, and artifact indexing
- `artifacts/transcripts`: immutable transcript artifacts and final-state snapshots
- `artifacts/deck_views`: immutable deck-view artifacts
- `artifacts/metric_results`: metric payloads and secondary outputs such as DTW alignment paths
- `artifacts/reports`: aggregated report outputs
- `artifacts/runtime_dbs`: per-trial cloned research DBs used by the harness

## Important Boundaries

- `main.py` remains the normal graph CLI entrypoint.
- `eval/graph_runner.py` exists only to reuse that same graph path from the eval harness.
- The harness never processes missing documents.
- The grader never creates missing deck views.
- The report stage never reruns grading.
- Benchmark-specific loading stays in the benchmark package.
- Metrics operate on deck views and source text, not raw graph internals.
