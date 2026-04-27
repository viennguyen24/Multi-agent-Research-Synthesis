from __future__ import annotations

import argparse
from _bootstrap import bootstrap_project_root

bootstrap_project_root()
from eval.config import load_eval_config
from eval.pipeline.grader import run_grader


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark grading over stored artifacts.")
    parser.add_argument("--config", default="eval/config.yaml")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--suite-id")
    args = parser.parse_args()
    config = load_eval_config(args.config)
    run_grader(config, args.benchmark, suite_id=args.suite_id)


if __name__ == "__main__":
    main()

