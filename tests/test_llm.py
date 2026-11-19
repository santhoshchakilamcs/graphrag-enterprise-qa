"""Tests for the optional LLM layer (llm.py).

The unit tests use a fake router (no model download). The one
integration test runs the real Qwen2.5-0.5B-Instruct model and is
skipped unless the model is already cached locally (no downloads in
tests) or GRAPHRAG_RUN_LLM_TESTS=1 is set.
"""

import os

import pytest

from graphrag_enterprise_qa.graph_build import build_graph
from graphrag_enterprise_qa.llm import (
    LLM_MODEL,
    _contains_answer,
    answer_question_llm,
    load_llm,
    validate_parse,
)
from graphrag_enterprise_qa.qa import UNKNOWN

LLM_TESTS_ENABLED = os.environ.get("GRAPHRAG_RUN_LLM_TESTS", "") in ("1", "true", "yes")


def _model_cached() -> bool:
    """True only if the model weights are fully downloaded."""
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(cache, "models--Qwen--Qwen2.5-0.5B-Instruct")
    snapshots = os.path.join(model_dir, "snapshots")
    if not os.path.isdir(snapshots):
        return False
    for snapshot in os.listdir(snapshots):
        files = os.listdir(os.path.join(snapshots, snapshot))
        if "model.safetensors" in files or "pytorch_model.bin" in files:
            return True
    return False


class FakeLLM:
    """A router that returns canned parses / compositions."""

    def __init__(self, parse_result, composed="The answer is X."):
        self.model_name = "fake-llm"
        self._parse_result = parse_result
        self._composed = composed
        self.parse_calls = 0
        self.compose_calls = 0

    def parse(self, question, idx):
        self.parse_calls += 1
        return self._parse_result

    def compose(self, question, answer, evidence):
        self.compose_calls += 1
        return self._composed


def test_validate_parse_accepts_good_output():
    _, idx = build_graph()
    parsed = validate_parse(
        {"intent": "revenue_category", "entities": {"category": ["Beverages"]}}, idx
    )
    assert parsed is not None
    assert parsed["intent"] == "revenue_category"
    assert parsed["entities"] == {"category": ["Beverages"]}


def test_validate_parse_accepts_bare_string_entities():
    # The model sometimes returns "category": "Beverages" instead of a list.
    _, idx = build_graph()
    parsed = validate_parse(
        {"intent": "revenue_category", "entities": {"category": "Beverages"}}, idx
    )
    assert parsed is not None
    assert parsed["entities"] == {"category": ["Beverages"]}


def test_validate_parse_rejects_unknown_intent():
    _, idx = build_graph()
    assert (
        validate_parse({"intent": "teleport_the_data", "entities": {}}, idx) is None
    )


def test_validate_parse_rejects_invented_entity():
    _, idx = build_graph()
    assert (
        validate_parse(
            {"intent": "revenue_category", "entities": {"category": ["Unobtainium"]}}, idx
        )
        is None
    )


def test_validate_parse_rejects_bad_shape():
    _, idx = build_graph()
    assert validate_parse({"intent": "revenue_category"}, idx) is None
    assert validate_parse("not a dict", idx) is None
    assert validate_parse(None, idx) is None


def test_llm_parse_drives_the_answer():
    graph, idx = build_graph()
    fake = FakeLLM(
        {"intent": "revenue_category", "entities": {"category": ["Beverages"]}, "years": [], "countries": []},
        composed="The total revenue from Beverages is $267,868.18.",
    )
    result = answer_question_llm(
        "What is the total revenue from the Beverages category?", graph, idx, fake
    )
    assert result["answer"] == "$267,868.18"
    assert result["parse_source"] == "llm"
    assert result["composed_contains_answer"] is True
    assert fake.parse_calls == 1
    assert fake.compose_calls == 1


def test_llm_fallback_when_parse_fails():
    graph, idx = build_graph()
    fake = FakeLLM(None)
    result = answer_question_llm(
        "What is the total revenue from the Beverages category?", graph, idx, fake
    )
    assert result["answer"] == "$267,868.18"
    assert result["parse_source"] == "fallback"


def test_llm_none_is_exactly_rule_based():
    graph, idx = build_graph()
    result = answer_question_llm(
        "Which employee handled the most orders in 1997?", graph, idx, None
    )
    assert result["answer"] == "Margaret Peacock"
    assert result["parse_source"] == "fallback"
    assert result["composed"] is None


def test_composed_sentence_does_not_change_the_score():
    graph, idx = build_graph()
    # The fake composes a wrong sentence; the scored answer must still
    # be the extractive graph value, and containment must report False.
    fake = FakeLLM(
        {"intent": "supplier_of_product", "entities": {"product": ["Chai"]}, "years": [], "countries": []},
        composed="The supplier is someone else entirely.",
    )
    result = answer_question_llm("Who is the supplier of the product Chai?", graph, idx, fake)
    assert result["answer"] == "Exotic Liquids"
    assert result["composed_contains_answer"] is False


def test_contains_answer_money_and_lists():
    assert _contains_answer("The total is $267,868.18.", "$267,868.18")
    assert _contains_answer("Revenue was 267868.18 dollars.", "$267,868.18")
    assert not _contains_answer("The total is $14,499.39.", "$267,868.18")
    assert _contains_answer("Products: A, B and C.", "A, B, C")
    assert not _contains_answer("Products: A only.", "A, B, C")


def test_unknown_question_composes_nothing():
    graph, idx = build_graph()
    fake = FakeLLM(None)
    result = answer_question_llm("What is the weather in Austin today?", graph, idx, fake)
    assert result["answer"] == UNKNOWN
    assert result["composed"] is None
    assert fake.compose_calls == 0


@pytest.mark.skipif(
    not (LLM_TESTS_ENABLED or _model_cached()),
    reason="real LLM test needs the cached Qwen2.5-0.5B-Instruct model",
)
def test_real_llm_routes_and_composes():
    llm = load_llm(LLM_MODEL)
    assert llm is not None
    graph, idx = build_graph()
    result = answer_question_llm(
        "What is the total revenue from the Beverages category?", graph, idx, llm
    )
    assert result["answer"] == "$267,868.18"
    assert result["parse_source"] in ("llm", "fallback")
    if result["composed"] is not None:
        assert isinstance(result["composed"], str) and result["composed"]
