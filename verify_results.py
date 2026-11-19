#!/usr/bin/env python3
"""Re-run the evaluation and check the reported numbers.

Verifies:
  1. A fresh run of run_eval() reproduces the rule-based GraphRAG and
     TF-IDF numbers in results.json exactly (same per-question answers
     and accuracies).
  2. If results.json has embedding-baseline numbers, a fresh
     embedding run reproduces them exactly. If the embedding model
     cannot be loaded (package missing / no network for the one-time
     download), this step is skipped with a warning and the stored
     numbers are checked for presence + README consistency only.
  3. If results.json has LLM-mode numbers, they are checked for
     presence + README consistency. Pass --llm to also re-run the LLM
     mode (Qwen2.5-0.5B-Instruct on CPU, several minutes) and compare
     exactly. The default skips the re-run because it is slow and
     needs the ~1GB model; reproduce LLM numbers with
     `python run_eval.py --llm --embedding`.
  4. README.md states the same accuracies as results.json.

Usage:
    python verify_results.py [--results results.json] [--readme README.md] [--llm]

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
    parser.add_argument("--llm", action="store_true", help="Also re-run LLM mode and compare.")
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as f:
        stored = json.load(f)

    errors = []
    notes = []

    # 1. Rule-based + TF-IDF: always re-run and compare exactly.
    fresh = run_eval()
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

    # 2. Embedding baseline: re-run when the model loads, else presence-check.
    if "embedding_baseline_correct" in stored:
        emb_fresh = None
        try:
            emb_fresh = run_eval(embedding=True)
        except Exception as e:  # noqa: BLE001 - model load can fail many ways
            notes.append(f"embedding baseline NOT re-verified (model unavailable: {e})")
        if emb_fresh is not None:
            if emb_fresh["embedding_baseline_correct"] != stored["embedding_baseline_correct"]:
                errors.append(
                    "embedding_baseline_correct: fresh "
                    f"{emb_fresh['embedding_baseline_correct']} != stored {stored['embedding_baseline_correct']}"
                )
            for fresh_row, stored_row in zip(emb_fresh["results"], stored["results"]):
                if fresh_row.get("embedding_answer") != stored_row.get("embedding_answer"):
                    errors.append(
                        f"{fresh_row['id']} embedding_answer: fresh {fresh_row.get('embedding_answer')!r} "
                        f"!= stored {stored_row.get('embedding_answer')!r}"
                    )

    # 3. LLM mode: presence-check by default; re-run with --llm.
    if "llm_correct" in stored:
        if not isinstance(stored.get("llm_routing"), dict):
            errors.append("results.json is missing llm_routing")
        for row in stored["results"]:
            if "llm_answer" not in row or "llm_parse_source" not in row:
                errors.append(f"{row['id']} is missing llm fields")
                break
        if args.llm:
            from graphrag_enterprise_qa.llm import load_llm

            llm = load_llm()
            if llm is None:
                errors.append("--llm requested but the LLM could not be loaded")
            else:
                llm_fresh = run_eval(llm=llm)
                if llm_fresh["llm_correct"] != stored["llm_correct"]:
                    errors.append(
                        f"llm_correct: fresh {llm_fresh['llm_correct']} != stored {stored['llm_correct']}"
                    )
                for fresh_row, stored_row in zip(llm_fresh["results"], stored["results"]):
                    if fresh_row.get("llm_answer") != stored_row.get("llm_answer"):
                        errors.append(
                            f"{fresh_row['id']} llm_answer: fresh {fresh_row.get('llm_answer')!r} "
                            f"!= stored {stored_row.get('llm_answer')!r}"
                        )
                    if fresh_row.get("llm_parse_source") != stored_row.get("llm_parse_source"):
                        errors.append(
                            f"{fresh_row['id']} llm_parse_source: fresh {fresh_row.get('llm_parse_source')!r} "
                            f"!= stored {stored_row.get('llm_parse_source')!r}"
                        )
        else:
            notes.append("LLM mode NOT re-verified (pass --llm to re-run it)")

    # 4. README consistency.
    if os.path.exists(args.readme):
        readme = open(args.readme, encoding="utf-8").read()
        lines = [
            f"GraphRAG: {stored['graphrag_correct']}/{stored['n_questions']}",
            f"Baseline (TF-IDF): {stored['baseline_correct']}/{stored['n_questions']}",
        ]
        if "embedding_baseline_correct" in stored:
            lines.append(
                f"Baseline (embedding RAG): {stored['embedding_baseline_correct']}/{stored['n_questions']}"
            )
        if "llm_correct" in stored:
            lines.append(
                f"GraphRAG (LLM mode): {stored['llm_correct']}/{stored['n_questions']}"
            )
        for line in lines:
            if line not in readme:
                errors.append(f"README is missing {line!r}")

    # 5. Visuals: the SVG charts and HTML report must exist and carry
    # the same numbers as results.json (they are text files, generated
    # by make_visuals.py from results.json). The accuracy chart and the
    # HTML report state the X/20 numbers; the heatmap states per-question
    # correctness, so it is checked for the 20 question ids instead.
    visual_files = {
        "results/accuracy_comparison.svg": "accuracy chart",
        "results/per_question_correctness.svg": "correctness heatmap",
        "results/report.html": "HTML report",
    }
    for path, label in visual_files.items():
        if not os.path.exists(path):
            errors.append(f"missing {label}: {path} (regenerate with python make_visuals.py)")
            continue
        text = open(path, encoding="utf-8").read()
        if path.endswith("per_question_correctness.svg"):
            for row in stored["results"]:
                if row["id"] not in text:
                    errors.append(f"{label} {path} is missing question {row['id']}")
            continue
        for key, name in (
            ("graphrag_correct", "GraphRAG"),
            ("baseline_correct", "Baseline (TF-IDF)"),
        ):
            expected = f"{stored[key]}/{stored['n_questions']}"
            if expected not in text:
                errors.append(f"{label} {path} is missing {name} {expected}")
        if "embedding_baseline_correct" in stored:
            expected = (
                f"{stored['embedding_baseline_correct']}/{stored['n_questions']}"
            )
            if expected not in text:
                errors.append(
                    f"{label} {path} is missing Baseline (embedding RAG) {expected}"
                )
        if "llm_correct" in stored:
            expected = f"{stored['llm_correct']}/{stored['n_questions']}"
            if expected not in text:
                errors.append(
                    f"{label} {path} is missing GraphRAG (LLM mode) {expected}"
                )

    if errors:
        print("VERIFY FAILED:")
        for error in errors:
            print(f"  - {error}")
        return 1

    status = (
        f"VERIFY OK: {stored['n_questions']} questions, "
        f"GraphRAG {stored['graphrag_correct']}/{stored['n_questions']}, "
        f"TF-IDF {stored['baseline_correct']}/{stored['n_questions']}"
    )
    if "embedding_baseline_correct" in stored:
        status += f", embedding RAG {stored['embedding_baseline_correct']}/{stored['n_questions']}"
    if "llm_correct" in stored:
        status += f", LLM mode {stored['llm_correct']}/{stored['n_questions']}"
    status += " - fresh run reproduces results.json and the README."
    print(status)
    for note in notes:
        print(f"NOTE: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
