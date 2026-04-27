from __future__ import annotations

import argparse
from _bootstrap import bootstrap_project_root

bootstrap_project_root()
from eval.config import load_eval_config
from eval.pipeline.common import init_eval_storage


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize eval storage and schema.")
    parser.add_argument("--config", default="eval/config.yaml")
    args = parser.parse_args()
    config = load_eval_config(args.config)
    with init_eval_storage(config):
        pass


if __name__ == "__main__":
    main()

