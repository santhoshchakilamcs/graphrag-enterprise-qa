"""Tests for the evaluation harness: reproducibility and honest scoring."""

import json
import os

from graphrag_enterprise_qa.eval import answers_match, run_eval

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results.json")


def test_eval_is_reproducible():
    first = run_eval()
    second = run_eval()
    assert first == second


def test_eval_counts():
    summary = run_eval()
    assert summary["n_questions"] == 20
    assert summary["graphrag_correct"] == 20
    assert summary["baseline_correct"] == 5
    assert summary["graphrag_accuracy"] == 1.0
    assert summary["baseline_accuracy"] == 0.25
    for row in summary["results"]:
        assert row["ground_truth"]
        assert isinstance(row["graphrag_correct"], bool)
        assert isinstance(row["baseline_correct"], bool)


def test_results_json_matches_fresh_run():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        stored = json.load(f)
    fresh = run_eval()
    assert stored["n_questions"] == fresh["n_questions"]
    assert stored["graphrag_correct"] == fresh["graphrag_correct"]
    assert stored["baseline_correct"] == fresh["baseline_correct"]
    for stored_row, fresh_row in zip(stored["results"], fresh["results"]):
        assert stored_row["ground_truth"] == fresh_row["ground_truth"]
        assert stored_row["graphrag_answer"] == fresh_row["graphrag_answer"]
        assert stored_row["baseline_answer"] == fresh_row["baseline_answer"]
    # The stored file also carries the embedding + LLM numbers, which
    # only a --llm --embedding run reproduces. Check they are present
    # and internally consistent (accuracies match correct/20).
    for key in (
        "embedding_baseline_correct",
        "embedding_baseline_accuracy",
        "llm_correct",
        "llm_accuracy",
    ):
        assert key in stored, f"results.json is missing {key}"
    assert (
        stored["embedding_baseline_accuracy"]
        == stored["embedding_baseline_correct"] / stored["n_questions"]
    )
    assert stored["llm_accuracy"] == stored["llm_correct"] / stored["n_questions"]
    assert stored["llm_routing"]["parsed_by_llm"] + stored["llm_routing"]["fallback_to_rules"] == 20
    for row in stored["results"]:
        assert "embedding_answer" in row and "embedding_correct" in row
        assert "llm_answer" in row and "llm_parse_source" in row


def test_answers_match_money_tolerance():
    assert answers_match("$234,507.29", "$234,507.29")
    assert answers_match("$1,525.05", "$1525.05")
    assert not answers_match("$100.00", "$101.00")


def test_answers_match_lists_are_order_insensitive():
    assert answers_match("B, A, C", "A, B, C")
    assert not answers_match("B, A", "A, B, C")
