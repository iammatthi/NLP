"""Aggregate completed Experiment 5 JSON reports into a README table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _mean_std(values: list[float]) -> str:
    if not values:
        return "pending"
    return f"{np.mean(values):.4f} +/- {np.std(values, ddof=1):.4f}" if len(values) > 1 else f"{values[0]:.4f}"


def collect(root: Path) -> list[dict]:
    rows = []
    for path in sorted(root.glob("*/eval_test.json")):
        if "_smoke" in path.parent.name:
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("n_total") is None or report.get("mode") != "argmax":
            continue
        rows.append(report)
    return rows


def build_readme(rows: list[dict]) -> str:
    groups: dict[tuple[str, str, int], list[dict]] = {}
    for row in rows:
        key = (row.get("model", "unknown"), row.get("condition", "unknown"), int(row.get("shots_per_class", 0)))
        groups.setdefault(key, []).append(row)
    lines = [
        "# Experiment 5: LLM Baselines",
        "",
        "Zero-/few-shot LLM severity classification on the shared Bengali test split.",
        "The headline comparison is `bn` versus `bn2en`; `en` is a sanity check only.",
        "",
        "The table below summarizes completed runs; smoke-test directories are excluded.",
        "",
        "## Results",
        "",
        "| model | condition | shots/class | n runs | coverage | macro-F1 | QWK | MAE | downgrade rate | upgrade rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (model, condition, shots), group in sorted(groups.items()):
        direction = [row["direction"] for row in group]
        lines.append(
            f"| `{model}` | `{condition}` | {shots} | {len(group)} | "
            f"{_mean_std([row['coverage'] for row in group])} | "
            f"{_mean_std([row['macro_f1'] for row in group])} | "
            f"{_mean_std([row['qwk'] for row in group])} | "
            f"{_mean_std([row['mae'] for row in group])} | "
            f"{_mean_std([item['downgrade_rate'] for item in direction])} | "
            f"{_mean_std([item['upgrade_rate'] for item in direction])} |"
        )
    if not groups:
        lines.append("| pending | pending | pending | 0 | pending | pending | pending | pending | pending | pending |")
    lines.extend([
        "",
        "## Protocol",
        "",
        """- Labels use the shared ordinal scale 1 minimal, 2 mild, 3 moderate, 4 severe.
- Few-shot examples are selected from training data only, balanced with k examples per class.
- Failed/refused answers are reported through coverage and a separate strict report.
- Raw post text is never written to result artefacts.
""",
        "## Limitations",
        "",
        "- LLM results, model IDs, and API availability must be recorded only after actual runs.",
        "- English is not a language-only comparison because its source dataset differs from Bengali.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/experiment5_llm"))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "README.md").write_text(build_readme(collect(args.root)), encoding="utf-8")


if __name__ == "__main__":
    main()