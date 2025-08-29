# backend/tests/conftest.py
from __future__ import annotations

import pytest

# ---------- Dummy embedder ----------
class DummyEmbedder:
    """Lightweight stand-in for SentenceTransformer."""
    def __init__(self, dims: int = 384):
        self.dims = dims

    def encode(self, texts, normalize_embeddings: bool = True):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for i, _ in enumerate(texts):
            vec = [0.0] * self.dims
            vec[i % self.dims] = 1.0  # simple deterministic non-zero
            out.append(vec)
        return out

def fake_sentence_transformer() -> DummyEmbedder:
    """
    Function your tests can import directly:
      from .conftest import fake_sentence_transformer
    """
    return DummyEmbedder()

@pytest.fixture
def embedder() -> DummyEmbedder:
    """
    Pytest fixture alternative:
      def test_something(embedder): ...
    """
    return DummyEmbedder()

# ---------- Fake Elasticsearch ----------
class _FakeIndices:
    def __init__(self):
        self._created = set()

    def exists(self, index: str) -> bool:
        return index in self._created

    def create(self, index: str, **_):
        self._created.add(index)

    def refresh(self, index: str):
        return

class FakeES:
    """
    Very small ES stub that supports the parts used by retrieve.py:
      - search() with BM25 (match), dense (knn), and ELSER (text_expansion)
      - indices.exists/create/refresh
      - ping()
    """
    def __init__(self, docs: list[dict] | None = None):
        self._indices = _FakeIndices()
        self._docs = docs or [
            {
                "_id": "1",
                "_source": {
                    "text": "Rule 13 — Overtaking. A vessel overtaking shall keep out of the way.",
                    "filename": "NavRules.pdf",
                    "chunk_id": "NavRules__0001",
                    "url": "https://example/navrules#r13",
                    "page": 42,
                    "heading": "Rule 13 — Overtaking",
                },
            },
            {
                "_id": "2",
                "_source": {
                    "text": "Rule 14 — Head-on situation.",
                    "filename": "NavRules.pdf",
                    "chunk_id": "NavRules__0002",
                    "url": "https://example/navrules#r14",
                    "page": 43,
                    "heading": "Rule 14 — Head-on",
                },
            },
            {   # NEW third doc so top_k=3 can succeed
                "_id": "3",
                "_source": {
                    "text": "Rule 15 — Crossing situation.",
                    "filename": "NavRules.pdf",
                    "chunk_id": "NavRules__0003",
                    "url": "https://example/navrules#r15",
                    "page": 44,
                    "heading": "Rule 15 — Crossing",
                },
            },
        ]

    @property
    def indices(self):
        return self._indices

    def ping(self) -> bool:
        return True

    def search(self, index: str, body: dict):
        size = body.get("size", 5)
        hits = []

        # Dense KNN
        if "knn" in body:
            sel = self._docs[: min(size, len(self._docs))]
            for i, d in enumerate(sel):
                hits.append({"_id": d["_id"], "_score": 1.0 / (i + 1), "_source": d["_source"]})

        # ELSER text_expansion
        elif body.get("query", {}).get("text_expansion"):
            sel = self._docs[: min(size, len(self._docs))]
            for i, d in enumerate(sel):
                hits.append({"_id": d["_id"], "_score": 0.9 / (i + 1), "_source": d["_source"]})

        # BM25 (match)
        else:
            sel = self._docs[: min(size, len(self._docs))]
            for i, d in enumerate(sel):
                hits.append({"_id": d["_id"], "_score": 0.8 / (i + 1), "_source": d["_source"]})

        return {"hits": {"hits": hits}}


@pytest.fixture
def fake_es() -> FakeES:
    return FakeES()
