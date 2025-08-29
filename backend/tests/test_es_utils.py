# backend/tests/test_es_utils.py
from backend.app.es_utils import _default_body

def test_default_mapping_has_fields():
    body = _default_body()
    props = body["mappings"]["properties"]

    # core fields
    assert props["text"]["type"] == "text"
    assert props["filename"]["type"] == "keyword"
    assert props["url"]["type"] == "keyword"
    assert props["chunk_id"]["type"] == "keyword"

    # vector fields
    assert props["embedding"]["type"] == "dense_vector"
    assert props["embedding"]["dims"] == 384
    assert props["ml.tokens"]["type"] == "sparse_vector"

    # metadata for UI/debug
    for meta in ("page", "heading", "section", "part_section"):
        assert meta in props
