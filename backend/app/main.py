from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

from .es_utils import get_es, ensure_index
from .retrieve import retrieve_docs
from .generate import generate_answer
from .guardrails import is_safe
from .ingest import ingest_local
from .drive import ingest_drive_url
from .elser_setup import setup_all

# Load .env (Windows-friendly)
load_dotenv(find_dotenv(), override=True)

# --------------------
# Config
# --------------------
ELASTIC_INDEX = os.getenv("ELASTIC_INDEX", "docs")
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")  # "elser" or "hybrid"
TOP_K = int(os.getenv("TOP_K", "5"))

DEFAULT_DRIVE_URL = os.getenv("GOOGLE_DRIVE_FILE_URL", "").strip()
AUTO_INGEST = os.getenv("AUTO_INGEST_ON_STARTUP", "false").lower() == "true"

# ELSER configuration
ELSER_AUTO_SETUP = os.getenv("ELSER_AUTO_SETUP", "true").lower() == "true"
ELSER_INFERENCE_ID = os.getenv("ELSER_INFERENCE_ID", "my-elser-endpoint")
ELSER_PIPELINE_ID = os.getenv("ELSER_PIPELINE_ID", "elser-v2-mltokens")

app = FastAPI(title="RAG with Elastic + Open LLM", version="0.3.0")


# --------------------
# Models
# --------------------
class QueryIn(BaseModel):
    question: str
    top_k: Optional[int] = None
    mode: Optional[str] = None  # "elser" or "hybrid"


class AnswerOut(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]


class IngestIn(BaseModel):
    folder: Optional[str] = None  # defaults to ./data/pdfs


class IngestDriveIn(BaseModel):
    url: Optional[str] = None      # file or folder Drive link
    limit: Optional[int] = None    # only first N PDFs when ingesting a folder


# --------------------
# Routes
# --------------------
@app.get("/")
def home():
    return {
        "ok": True,
        "index": ELASTIC_INDEX,
        "elser_auto_setup": ELSER_AUTO_SETUP,
        "endpoints": [
            "/healthz",
            "/query",
            "/query_debug",
            "/ingest",
            "/ingest_drive",
            "/setup_elser",
            "/docs",
        ],
    }


@app.get("/healthz")
def healthz():
    try:
        es = get_es()
        ok = es.ping()
        return {"status": "ok", "elasticsearch": ok}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.post("/query", response_model=AnswerOut)
def query(body: QueryIn):
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question.")
    if not is_safe(question):
        raise HTTPException(status_code=400, detail="Unsafe or disallowed query.")

    es = get_es()
    ensure_index(es, ELASTIC_INDEX)

    mode = (body.mode or RETRIEVAL_MODE).lower()
    k = int(body.top_k or TOP_K)

    docs = retrieve_docs(es, ELASTIC_INDEX, question, mode=mode, top_k=k)
    if not docs:
        return {"answer": "I don’t know.", "citations": []}

    answer = generate_answer(question, docs)

    citations: List[Dict[str, Any]] = []
    for d in docs:
        citations.append({
            "title": d.get("filename", "document"),
            "link": d.get("url"),
            "snippet": (d.get("text", "") or "")[:300],
            "page": d.get("page"),
            "heading": d.get("heading"),
            "section": d.get("section"),
            "part_section": d.get("part_section"),
            "chunk_id": d.get("chunk_id"),
            "source": d.get("source"),   # bm25 | dense | elser (first source seen)
            "rrf": d.get("rrf"),         # fused score
        })

    return {"answer": answer, "citations": citations}


# Raw matches for debugging / “where did it come from?”
@app.post("/query_debug")
def query_debug(body: QueryIn):
    question = (body.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question.")
    if not is_safe(question):
        raise HTTPException(status_code=400, detail="Unsafe or disallowed query.")

    es = get_es()
    ensure_index(es, ELASTIC_INDEX)

    mode = (body.mode or RETRIEVAL_MODE).lower()
    k = int(body.top_k or TOP_K)

    docs = retrieve_docs(es, ELASTIC_INDEX, question, mode=mode, top_k=k)
    hits = []
    for d in docs:
        hits.append({
            "filename": d.get("filename"),
            "url": d.get("url"),
            "chunk_id": d.get("chunk_id"),
            "page": d.get("page"),
            "heading": d.get("heading"),
            "section": d.get("section"),
            "part_section": d.get("part_section"),
            "score": d.get("score"),
            "rrf": d.get("rrf"),
            "snippet": (d.get("text", "") or "")[:300],
        })
    return {"hits": hits, "count": len(hits)}


@app.post("/ingest")
def ingest(body: IngestIn = Body(default=None)):
    folder = (body.folder if body and body.folder else "data/pdfs").strip()
    if not os.path.isdir(folder):
        raise HTTPException(status_code=400, detail=f"Folder not found: {folder}")
    es = get_es()
    ensure_index(es, ELASTIC_INDEX)
    # NOTE: ingest_local should call index_docs(..., pipeline=ELSER_PIPELINE_ID) internally
    count = ingest_local(es, ELASTIC_INDEX, folder, pipeline=ELSER_PIPELINE_ID)
    return {"indexed": count, "index": ELASTIC_INDEX, "folder": folder, "pipeline": ELSER_PIPELINE_ID}


@app.post("/ingest_drive")
def ingest_drive(body: IngestDriveIn = Body(default=None)):
    url = (body.url.strip() if body and body.url else DEFAULT_DRIVE_URL)
    if not url:
        raise HTTPException(status_code=400, detail="No Drive URL provided and GOOGLE_DRIVE_FILE_URL not set.")
    es = get_es()
    ensure_index(es, ELASTIC_INDEX)
    try:
        count = ingest_drive_url(
            es, ELASTIC_INDEX, url,
            pipeline=ELSER_PIPELINE_ID,
            limit=(body.limit if body else None),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ingest failed: {e}")
    return {"indexed": count, "index": ELASTIC_INDEX, "url": url, "pipeline": ELSER_PIPELINE_ID}


@app.post("/setup_elser")
def setup_elser():
    out = setup_all(index=ELASTIC_INDEX, do_trial=True, do_backfill=True)
    return {"ok": True, **out}


# --------------------
# Startup hooks
# --------------------
@app.on_event("startup")
def on_startup():
    es = get_es()
    ensure_index(es, ELASTIC_INDEX)

    if ELSER_AUTO_SETUP:
        out = setup_all(index=ELASTIC_INDEX, do_trial=True, do_backfill=True)
        print(f"[ELSER setup] {out}")

    # Optional: auto-ingest one Drive link at boot
    if AUTO_INGEST and DEFAULT_DRIVE_URL:
        try:
            n = ingest_drive_url(es, ELASTIC_INDEX, DEFAULT_DRIVE_URL, pipeline=ELSER_PIPELINE_ID, limit=None)
            print(f"[startup] auto-ingest complete: {n} chunks")
        except Exception as e:
            print(f"[startup] auto-ingest failed: {e}")
