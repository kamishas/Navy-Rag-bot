# backend/app/ingest.py
from __future__ import annotations

import os
import re
from typing import List, Dict, Optional, Tuple

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from .es_utils import index_docs

# ---- Embedding model (configurable via ENV) ----
_EMB = SentenceTransformer(os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))

# -------- PDF reading helpers --------
def read_pdf_pages(path: str) -> List[str]:
    """Return list of page texts (1-based page numbers correspond to index+1)."""
    reader = PdfReader(path)
    pages: List[str] = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return pages

def read_pdf_text(path: str) -> str:
    """Whole-document text (kept for backwards compat)."""
    return "\n".join(read_pdf_pages(path)).strip()

# -------- chunking / metadata helpers --------
def chunk_words(words: List[str], target_words: int, overlap_words: int) -> List[List[str]]:
    chunks: List[List[str]] = []
    i = 0
    step = max(1, target_words - overlap_words)
    while i < len(words):
        seg = words[i : i + target_words]
        if not seg:
            break
        chunks.append(seg)
        i += step
    return chunks

# Generic heading regex (works for any rule number)
_HEADING_PAT = re.compile(r"(?:\bRule\s*\d+\b[^\n]*|\bOvertaking\b)", re.IGNORECASE)
_SECTION_PAT = re.compile(r"(?:\bINTERNATIONAL\b|\bINLAND\b)", re.IGNORECASE)
_PART_SEC_PAT = re.compile(r"(?:Part\s+[A-Z]\b[^\n]*|Section\s+[IVX]+\b[^\n]*)", re.IGNORECASE)

def _find_heading_and_section(page_text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Heuristics to extract heading/section lines from a page."""
    heading = None
    section = None
    partsec = None

    for line in page_text.splitlines():
        if _HEADING_PAT.search(line):
            heading = line.strip()
            break

    for line in page_text.splitlines():
        if _SECTION_PAT.search(line):
            section = line.strip()
            break

    for line in page_text.splitlines():
        if _PART_SEC_PAT.search(line):
            partsec = line.strip()
            break

    return heading, section, partsec

def chunk_text(text: str, target_words: int = 320, overlap_words: int = 60) -> List[str]:
    """Sliding-window chunking across the whole string."""
    words = (text or "").split()
    chunks = chunk_words(words, target_words, overlap_words)
    return [" ".join(c) for c in chunks]

def embed(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    return _EMB.encode(texts, normalize_embeddings=True).tolist()

# -------- local ingest --------
def build_docs_from_folder(folder: str) -> List[Dict]:
    """
    Parse PDFs page-by-page, chunk within each page so we can attach page + heading/section metadata.
    """
    docs: List[Dict] = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if not fn.lower().endswith(".pdf"):
                continue

            fpath = os.path.join(root, fn)
            pages = read_pdf_pages(fpath)
            if not pages:
                continue

            for p_idx, ptxt in enumerate(pages, start=1):  # 1-based page number
                heading, section, partsec = _find_heading_and_section(ptxt)

                # Build all chunks for this page and embed them in one go (faster)
                words = (ptxt or "").split()
                page_chunks = [" ".join(cw) for cw in chunk_words(words, 320, 60)]
                page_embs = embed(page_chunks)

                for c_idx, (chunk, vec) in enumerate(zip(page_chunks, page_embs)):
                    docs.append(
                        {
                            "text": chunk,
                            "filename": fn,
                            "url": f"file://{fpath.replace(os.sep, '/')}",
                            "chunk_id": f"{fn}__p{p_idx:03d}_{c_idx:02d}",
                            "embedding": vec,
                            "page": p_idx,
                            "heading": heading,
                            "section": section,
                            "part_section": partsec,
                        }
                    )
    return docs

def ingest_local(es, index: str, folder: str, pipeline: Optional[str] = None) -> int:
    docs = build_docs_from_folder(folder)
    return index_docs(es, index, docs, pipeline=pipeline)
