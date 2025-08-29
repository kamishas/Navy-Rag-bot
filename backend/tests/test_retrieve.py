# backend/tests/test_retrieve.py
from .conftest import FakeES  # ✅ now exists
from backend.app.retrieve import _rrf, retrieve_docs

def test_rrf_merges_and_ranks():
    b1 = [{"filename":"a","chunk_id":"x"}, {"filename":"b","chunk_id":"y"}]  # ranks 1,2
    b2 = [{"filename":"b","chunk_id":"y"}, {"filename":"c","chunk_id":"z"}]
    fused = _rrf([b1, b2], k=60)
    # Should include all unique keys, keep order by fused score
    keys = [(d["filename"], d["chunk_id"]) for d in fused]
    assert set(keys) == {("a","x"),("b","y"),("c","z")}
    # 'b' appears twice, should be ranked first
    assert keys[0] == ("b","y")

def test_retrieve_docs_hybrid_topk_and_rrf_present():
    es = FakeES()
    docs = retrieve_docs(es, index="docs", question="overtaking", mode="hybrid", top_k=3)
    assert len(docs) == 3
    # rrf score added
    assert all("rrf" in d for d in docs)
    # includes fields from fake hits
    assert any(d["source"] == "elser" or d["text"].startswith(("elser","dense","bm25")) for d in docs)
