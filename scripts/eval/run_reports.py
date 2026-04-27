from __future__ import annotations

import argparse
from _bootstrap import bootstrap_project_root

bootstrap_project_root()
from eval.config import load_eval_config
from eval.pipeline.reports import run_reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aggregate eval reports from stored metric results.")
    parser.add_argument("--config", default="eval/config.yaml")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--suite-id")
    args = parser.parse_args()
    config = load_eval_config(args.config)
    run_reports(config, args.benchmark, suite_id=args.suite_id)


if __name__ == "__main__":
    main()

