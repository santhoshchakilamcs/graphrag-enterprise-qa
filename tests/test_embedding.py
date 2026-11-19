"""Tests for the embedding RAG baseline (embedding_baseline.py).

Unit tests use a stub encoder (no model download). The integration
test runs the real all-MiniLM-L6-v2 model and is skipped unless it is
already cached locally (no downloads in tests) or
GRAPHRAG_RUN_EMBEDDING_TESTS=1 is set.
"""

import os
import sys
import types

import numpy as np
import pytest

from graphrag_enterprise_qa.embedding_baseline import EMBEDDING_MODEL, EmbeddingBaseline
from graphrag_enterprise_qa.graph_build import build_graph

EMBEDDING_TESTS_ENABLED = os.environ.get("GRAPHRAG_RUN_EMBEDDING_TESTS", "") in (
    "1",
    "true",
    "yes",
)


def _model_cached() -> bool:
    """True only if the model weights are fully downloaded."""
    cache = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(cache, "models--sentence-transformers--all-MiniLM-L6-v2")
    snapshots = os.path.join(model_dir, "snapshots")
    if not os.path.isdir(snapshots):
        return False
    for snapshot in os.listdir(snapshots):
        files = os.listdir(os.path.join(snapshots, snapshot))
        if "model.safetensors" in files or "pytorch_model.bin" in files:
            return True
    return False


STOPWORDS = frozenset(
    "a an and are as at be by did do does for how in is it most of on or the "
    "to what which who whose with".split()
)


class FakeEncoder:
    """Deterministic exact-word embeddings (unit tests only).

    Each distinct content word gets its own dimension, so retrieval
    ranks on shared words with no hash collisions. Stopwords dropped.
    """

    def __init__(self, *args, **kwargs):
        self.vocab: dict[str, int] = {}
        self._frozen = False

    def _words(self, text):
        for word in text.lower().split():
            word = word.strip(".,:;()")
            if word and word not in STOPWORDS:
                yield word

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        # The vocabulary is fixed by the first call (the chunk batch);
        # later calls (queries) ignore words outside it, so all vectors
        # share one dimension.
        if not self._frozen:
            for text in texts:
                for word in self._words(text):
                    if word not in self.vocab:
                        self.vocab[word] = len(self.vocab)
            self._frozen = True
        dim = len(self.vocab)
        vecs = []
        for text in texts:
            v = np.zeros(dim)
            for word in self._words(text):
                if word in self.vocab:
                    v[self.vocab[word]] += 1.0
            norm = np.linalg.norm(v)
            if normalize_embeddings and norm > 0:
                v = v / norm
            vecs.append(v)
        return np.array(vecs)


@pytest.fixture
def stub_sentence_transformers(monkeypatch):
    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = FakeEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub)
    return stub


def test_embedding_baseline_uses_same_chunks(stub_sentence_transformers):
    baseline = EmbeddingBaseline()
    assert len(baseline.chunks) == 3202
    assert baseline.embeddings.shape[0] == 3202
    assert baseline.embeddings.shape[1] > 100


def test_embedding_retrieve_returns_top_k(stub_sentence_transformers):
    baseline = EmbeddingBaseline()
    retrieved = baseline.retrieve("Which supplier provides the most products in the Beverages category?")
    assert 0 < len(retrieved) <= 50
    assert any("beverages" in chunk.lower() for chunk in retrieved[:5])


def test_embedding_answers_a_lookup(stub_sentence_transformers):
    _, idx = build_graph()
    baseline = EmbeddingBaseline()
    answer, _ = baseline.answer_question("Who is the supplier of the product Chai?", idx)
    assert answer == "Exotic Liquids"


def test_embedding_is_deterministic(stub_sentence_transformers):
    _, idx = build_graph()
    baseline = EmbeddingBaseline()
    first = baseline.answer_question("How many customers are based in Germany?", idx)
    second = baseline.answer_question("How many customers are based in Germany?", idx)
    assert first == second


@pytest.mark.skipif(
    not (EMBEDDING_TESTS_ENABLED or _model_cached()),
    reason="real embedding test needs the cached all-MiniLM-L6-v2 model",
)
def test_real_embedding_baseline_answers_a_lookup():
    _, idx = build_graph()
    baseline = EmbeddingBaseline(model_name=EMBEDDING_MODEL)
    answer, _ = baseline.answer_question("Who is the supplier of the product Chai?", idx)
    assert answer == "Exotic Liquids"
