#!/usr/bin/env python3
"""Run the full evaluation and write results.json.

Usage:
    python run_eval.py [--data-dir data] [--llm] [--embedding]

By default this runs the fast path only: rule-based GraphRAG and the
TF-IDF baseline. --embedding adds the dense embedding baseline
(sentence-transformers, downloads ~90MB once). --llm adds LLM-mode
GraphRAG (Qwen2.5-0.5B-Instruct on CPU, downloads ~1GB once; takes
several minutes). The numbers reported in the README come from:

    python run_eval.py --llm --embedding

A run without the flags writes results.json without the extra keys;
use the full command above to reproduce the reported numbers.
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
    parser = argparse.ArgumentParser(description="Run the GraphRAG evaluation.")
    parser.add_argument("--data-dir", default=None, help="Directory holding the Northwind CSVs.")
    parser.add_argument("--out", default="results.json", help="Where to write the results JSON.")
    parser.add_argument(
        "--llm", action="store_true", help="Also run LLM-mode GraphRAG (Qwen2.5-0.5B-Instruct)."
    )
    parser.add_argument(
        "--embedding", action="store_true", help="Also run the embedding RAG baseline."
    )
    args = parser.parse_args()

    graph, _ = build_graph(args.data_dir)
    stats = graph_stats(graph)
    print(f"Graph: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
    for node_type, count in stats["nodes"].items():
        print(f"  {node_type}: {count}")
    for edge_type, count in stats["edges"].items():
        print(f"  {edge_type}: {count}")

    llm = None
    if args.llm:
        from graphrag_enterprise_qa.llm import load_llm

        llm = load_llm()
        if llm is None:
            print("ERROR: --llm requested but the LLM could not be loaded; not writing results.")
            return 1

    summary = run_eval(args.data_dir, llm=llm, embedding=args.embedding)
    summary["graph_stats"] = stats

    print()
    for row in summary["results"]:
        mark_g = "OK " if row["graphrag_correct"] else "MISS"
        mark_b = "OK " if row["baseline_correct"] else "MISS"
        line = (
            f"{row['id']:<5}{row['question'][:60]:<62}{str(row['ground_truth'])[:26]:<28}"
            f"{mark_g} {str(row['graphrag_answer'])[:22]:<25}{mark_b} {str(row['baseline_answer'])[:22]}"
        )
        if "embedding_answer" in row:
            mark_e = "OK " if row["embedding_correct"] else "MISS"
            line += f"{mark_e} {str(row['embedding_answer'])[:22]:<25}"
        if "llm_answer" in row:
            mark_l = "OK " if row["llm_correct"] else "MISS"
            src = row["llm_parse_source"][:4]
            line += f"{mark_l} {str(row['llm_answer'])[:20]:<22}({src})"
        print(line)

    print()
    print(f"GraphRAG:  {summary['graphrag_correct']}/{summary['n_questions']} "
          f"= {summary['graphrag_accuracy']:.1%}")
    print(f"Baseline (TF-IDF):  {summary['baseline_correct']}/{summary['n_questions']} "
          f"= {summary['baseline_accuracy']:.1%}")
    if "embedding_baseline_correct" in summary:
        print(f"Baseline (embedding RAG):  {summary['embedding_baseline_correct']}/{summary['n_questions']} "
              f"= {summary['embedding_baseline_accuracy']:.1%}")
    if "llm_correct" in summary:
        print(f"GraphRAG (LLM mode):  {summary['llm_correct']}/{summary['n_questions']} "
              f"= {summary['llm_accuracy']:.1%}")
        print(f"  LLM routing: {summary['llm_routing']['parsed_by_llm']} parsed by the LLM, "
              f"{summary['llm_routing']['fallback_to_rules']} fell back to rules")
        print(f"  Composed answers stating the value: "
              f"{summary['llm_composed_contains_answer']}/{summary['n_questions']}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {args.out}")
    print("Regenerate the charts and report: python make_visuals.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
