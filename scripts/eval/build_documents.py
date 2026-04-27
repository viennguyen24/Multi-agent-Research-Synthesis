from __future__ import annotations

import argparse
from _bootstrap import bootstrap_project_root

bootstrap_project_root()
from eval.config import load_eval_config
from eval.pipeline.documents import build_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Build processed documents for eval tasks.")
    parser.add_argument("--config", default="eval/config.yaml")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--suite-id")
    parser.add_argument("--task-id", action="append", dest="task_ids")
    args = parser.parse_args()
    config = load_eval_config(args.config)
    build_documents(config, args.benchmark, suite_id=args.suite_id, task_ids=args.task_ids)


if __name__ == "__main__":
    main()

