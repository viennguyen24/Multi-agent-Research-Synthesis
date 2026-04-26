from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from eval.config import EvalConfig
from eval.pipeline.common import init_eval_storage


def run_reports(
    config: EvalConfig,
    benchmark_id: str,
    suite_id: str | None = None,
) -> str:
    with init_eval_storage(config) as db:
        metric_rows = db.list_metric_results(benchmark_id=benchmark_id, suite_id=suite_id)

    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in metric_rows:
        if row["scalar_value"] is None:
            continue
        key = (
            row["metric_id"],
            row["variant_id"],
            row["graph_version"],
            row["doc_pipeline_version"],
        )
        grouped[key].append(float(row["scalar_value"]))

    output_path = config.paths.reports_dir / f"{benchmark_id}_{suite_id or 'all'}_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "metric_id",
            "variant_id",
            "graph_version",
            "doc_pipeline_version",
            "count",
            "mean",
        ])
        for key in sorted(grouped):
            values = grouped[key]
            writer.writerow([*key, len(values), sum(values) / len(values)])
    return str(output_path)
