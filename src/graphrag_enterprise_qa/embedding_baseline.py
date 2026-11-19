"""Embedding-based RAG baseline (sentence-transformers, all-MiniLM-L6-v2).

Same chunks, same top-k, same answer handlers as the TF-IDF baseline
in baseline.py - the only difference is retrieval: dense embedding
cosine similarity instead of TF-IDF. So the comparison isolates
retrieval quality (sparse vs dense) while everything else is held
constant.

The model downloads at runtime into the Hugging Face cache
(~/.cache/huggingface); nothing is stored in the repo. The import is
lazy (inside __init__) so the rest of the package works without
sentence-transformers installed.
"""

from __future__ import annotations

import numpy as np

from .baseline import TOP_K, Baseline, build_chunks

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingBaseline(Baseline):
    """Dense retriever + the same extraction handlers as the TF-IDF baseline."""

    def __init__(
        self,
        data_dir: str | None = None,
        top_k: int = TOP_K,
        model_name: str = EMBEDDING_MODEL,
    ):
        self.chunks = build_chunks(data_dir)
        self.top_k = top_k
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is not installed; pip install -r requirements.txt"
            ) from e
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            self.chunks, normalize_embeddings=True, show_progress_bar=False
        )

    def retrieve(self, question: str) -> list[str]:
        query = self.model.encode(
            [question], normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores = self.embeddings @ query
        order = np.argsort(-scores, kind="stable")[: self.top_k]
        return [self.chunks[i] for i in order if scores[i] > 0]
