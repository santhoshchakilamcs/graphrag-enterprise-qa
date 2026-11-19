"""Tests for make_visuals.py: SVGs and the HTML report are UTF-8 text,
valid XML/HTML, and carry the numbers from results.json."""

import html
import os
import sys
import xml.etree.ElementTree as ET

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import make_visuals  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("visuals")
    original_out_dir = make_visuals.OUT_DIR
    make_visuals.OUT_DIR = str(out_dir)
    try:
        results_path = os.path.join(
            os.path.dirname(__file__), "..", "results.json"
        )
        rc = make_visuals.main([results_path])
        assert rc == 0
        yield out_dir
    finally:
        make_visuals.OUT_DIR = original_out_dir


def test_visual_files_are_utf8_text(built):
    for name in (
        "accuracy_comparison.svg",
        "per_question_correctness.svg",
        "report.html",
    ):
        path = os.path.join(str(built), name)
        assert os.path.exists(path), f"missing {name}"
        raw = open(path, "rb").read()
        assert b"\x00" not in raw, f"{name} contains NUL bytes (not text)"
        text = raw.decode("utf-8")
        assert len(text) > 500, f"{name} looks empty"


def test_svgs_are_valid_xml(built):
    for name in ("accuracy_comparison.svg", "per_question_correctness.svg"):
        path = os.path.join(str(built), name)
        tree = ET.parse(path)
        assert tree.getroot().tag.endswith("svg"), f"{name} root is not <svg>"


def test_accuracy_chart_carries_the_numbers(built):
    summary = make_visuals.load_results()
    text = open(
        os.path.join(str(built), "accuracy_comparison.svg"), encoding="utf-8"
    ).read()
    for label in ("GraphRAG (rule-based)", "Baseline (TF-IDF)"):
        assert label in text
    assert f"{summary['graphrag_correct']}/{summary['n_questions']}" in text
    assert f"{summary['baseline_correct']}/{summary['n_questions']}" in text


def test_html_report_has_systems_and_questions(built):
    text = open(
        os.path.join(str(built), "report.html"), encoding="utf-8"
    ).read()
    summary = make_visuals.load_results()
    for label in ("GraphRAG (rule-based)", "Baseline (TF-IDF)"):
        assert label in text
    # Every question id and ground truth appears in the per-question table
    # (HTML-escaped: apostrophes become &#x27;).
    for row in summary["results"]:
        assert row["id"] in text
        assert html.escape(row["ground_truth"]) in text
    # The SVGs are inlined, not referenced as external files.
    assert "<svg" in text
    assert 'src="http' not in text
    assert 'href="http' not in text
    assert "<img" not in text


def test_visuals_match_results_json(built):
    """The chart and the report agree with results.json exactly."""
    summary = make_visuals.load_results()
    html_text = open(
        os.path.join(str(built), "report.html"), encoding="utf-8"
    ).read()
    assert f"{summary['graphrag_correct']}/{summary['n_questions']}" in html_text
    assert f"{summary['baseline_correct']}/{summary['n_questions']}" in html_text
    if "llm_correct" in summary:
        assert f"{summary['llm_correct']}/{summary['n_questions']}" in html_text
    if "embedding_baseline_correct" in summary:
        assert (
            f"{summary['embedding_baseline_correct']}/"
            f"{summary['n_questions']}"
        ) in html_text
