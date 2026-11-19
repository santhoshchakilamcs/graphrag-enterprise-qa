#!/usr/bin/env python3
"""Run the full evaluation and write results.json.

Usage:
    python run_eval.py [--data-dir data]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from graphrag_enterprise_qa.eval import run_eval  # noqa: E402
from graphrag_enterprise_qa.graph_build import build_graph, graph_stats  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the GraphRAG vs TF-IDF evaluation.")
    parser.add_argument("--data-dir", default=None, help="Directory holding the Northwind CSVs.")
    parser.add_argument("--out", default="results.json", help="Where to write the results JSON.")
    args = parser.parse_args()

    graph, _ = build_graph(args.data_dir)
    stats = graph_stats(graph)
    print(f"Graph: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
    for node_type, count in stats["nodes"].items():
        print(f"  {node_type}: {count}")
    for edge_type, count in stats["edges"].items():
        print(f"  {edge_type}: {count}")

    summary = run_eval(args.data_dir)
    summary["graph_stats"] = stats

    print()
    print(f"{'ID':<5}{'Question':<62}{'GT':<28}{'GraphRAG':<28}{'Baseline':<28}")
    for row in summary["results"]:
        mark_g = "OK " if row["graphrag_correct"] else "MISS"
        mark_b = "OK " if row["baseline_correct"] else "MISS"
        print(f"{row['id']:<5}{row['question'][:60]:<62}{str(row['ground_truth'])[:26]:<28}"
              f"{mark_g} {str(row['graphrag_answer'])[:22]:<25}{mark_b} {str(row['baseline_answer'])[:22]}")

    print()
    print(f"GraphRAG:  {summary['graphrag_correct']}/{summary['n_questions']} "
          f"= {summary['graphrag_accuracy']:.1%}")
    print(f"Baseline:  {summary['baseline_correct']}/{summary['n_questions']} "
          f"= {summary['baseline_accuracy']:.1%}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
