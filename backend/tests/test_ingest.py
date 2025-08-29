# backend/tests/test_ingest.py
from .conftest import fake_sentence_transformer  # ✅ now exists
# backend/tests/test_ingest.py
from backend.app.ingest import (
    chunk_words,
    chunk_text,              # ← add this
    _find_heading_and_section,
)


def test_chunk_words_overlap():
    words = [f"w{i}" for i in range(20)]
    chunks = chunk_words(words, target_words=10, overlap_words=2)
    # Windows of length 10, step = 8 → 3 chunks (0..9), (8..17), (16..19)
    assert len(chunks) == 3
    # Overlap check: last 2 of first == first 2 of second
    assert chunks[0][-2:] == chunks[1][:2]

def test_chunk_text_simple():
    txt = " ".join([f"w{i}" for i in range(50)])
    parts = chunk_text(txt, target_words=20, overlap_words=5)
    assert len(parts) >= 2
    assert isinstance(parts[0], str)

def test_heading_extraction_generic():
    page = "Part B — Steering and Sailing Rules\nINTERNATIONAL\nRule 13 — Overtaking\nSome body text."
    heading, section, partsec = _find_heading_and_section(page)
    assert "Rule 13" in (heading or "")
    assert section == "INTERNATIONAL"
    assert "Part B" in (partsec or "")
