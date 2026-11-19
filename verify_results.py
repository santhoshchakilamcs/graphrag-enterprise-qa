#!/usr/bin/env python3
"""Re-run the evaluation and check the reported numbers.

Verifies:
  1. A fresh run of run_eval() reproduces results.json exactly
     (same per-question answers and accuracies).
  2. README.md states the same accuracies as results.json.

Usage:
    python verify_results.py [--results results.json] [--readme README.md]

Exits non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from graphrag_enterprise_qa.eval import run_eval  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify reported evaluation numbers.")
    parser.add_argument("--results", default="results.json")
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as f:
        stored = json.load(f)

    fresh = run_eval()
    errors = []

    if fresh["n_questions"] != stored["n_questions"]:
        errors.append(f"n_questions: fresh {fresh['n_questions']} != stored {stored['n_questions']}")
    if fresh["graphrag_correct"] != stored["graphrag_correct"]:
        errors.append(f"graphrag_correct: fresh {fresh['graphrag_correct']} != stored {stored['graphrag_correct']}")
    if fresh["baseline_correct"] != stored["baseline_correct"]:
        errors.append(f"baseline_correct: fresh {fresh['baseline_correct']} != stored {stored['baseline_correct']}")
    for fresh_row, stored_row in zip(fresh["results"], stored["results"]):
        for key in ("ground_truth", "graphrag_answer", "baseline_answer"):
            if fresh_row[key] != stored_row[key]:
                errors.append(
                    f"{fresh_row['id']} {key}: fresh {fresh_row[key]!r} != stored {stored_row[key]!r}"
                )

    if os.path.exists(args.readme):
        readme = open(args.readme, encoding="utf-8").read()
        graph_line = f"GraphRAG: {stored['graphrag_correct']}/{stored['n_questions']}"
        base_line = f"Baseline (TF-IDF): {stored['baseline_correct']}/{stored['n_questions']}"
        if graph_line not in readme:
            errors.append(f"README is missing {graph_line!r}")
        if base_line not in readme:
            errors.append(f"README is missing {base_line!r}")

    if errors:
        print("VERIFY FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(
        f"VERIFY OK: {stored['n_questions']} questions, "
        f"GraphRAG {stored['graphrag_correct']}/{stored['n_questions']}, "
        f"baseline {stored['baseline_correct']}/{stored['n_questions']} - "
        "fresh run reproduces results.json and the README."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
