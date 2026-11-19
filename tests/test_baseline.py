"""Tests for the TF-IDF retrieval baseline."""

from graphrag_enterprise_qa.baseline import Baseline, build_chunks
from graphrag_enterprise_qa.graph_build import build_graph


def test_chunk_count():
    chunks = build_chunks()
    # 8 categories + 91 customers + 29 suppliers + 9 employees + 3 shippers
    # + 77 products + 830 orders + 2155 order details
    assert len(chunks) == 3202


def test_retrieval_returns_top_k():
    baseline = Baseline()
    retrieved = baseline.retrieve("Which supplier provides the most products in the Beverages category?")
    assert 0 < len(retrieved) <= 50
    # The top hits should be relevant: at least one mentions Beverages.
    assert any("beverages" in chunk.lower() for chunk in retrieved[:5])


def test_baseline_answers_a_lookup():
    _, idx = build_graph()
    baseline = Baseline()
    answer, _ = baseline.answer_question("Who is the supplier of the product Chai?", idx)
    assert answer == "Exotic Liquids"


def test_baseline_is_deterministic():
    _, idx = build_graph()
    baseline = Baseline()
    first = baseline.answer_question("How many customers are based in Germany?", idx)
    second = baseline.answer_question("How many customers are based in Germany?", idx)
    assert first == second
