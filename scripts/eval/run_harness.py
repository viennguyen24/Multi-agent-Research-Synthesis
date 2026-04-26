from __future__ import annotations

import argparse

from eval.config import load_eval_config
from eval.pipeline.harness import run_harness


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval harness on normalized tasks.")
    parser.add_argument("--config", default="eval/config.yaml")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--variant-id", required=True)
    parser.add_argument("--graph-version", required=True)
    parser.add_argument("--doc-pipeline-version", required=True)
    parser.add_argument("--trial-count", type=int, default=1)
    parser.add_argument("--suite-id")
    parser.add_argument("--task-id", action="append", dest="task_ids")
    args = parser.parse_args()
    config = load_eval_config(args.config)
    run_harness(
        config=config,
        benchmark_id=args.benchmark,
        variant_id=args.variant_id,
        graph_version=args.graph_version,
        doc_pipeline_version=args.doc_pipeline_version,
        trial_count=args.trial_count,
        suite_id=args.suite_id,
        task_ids=args.task_ids,
    )


if __name__ == "__main__":
    main()

