from __future__ import annotations

import os
from typing import List, Dict, Any
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

_EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

ELSER_INFERENCE_ID = os.getenv("ELSER_INFERENCE_ID", "my-elser-endpoint")
# If you installed ELSER as a model instead of an inference endpoint, set this:
ELSER_MODEL_ID = os.getenv("ELSER_MODEL_ID", ".elser_model_2_linux-x86_64")

def _bm25(es: Elasticsearch, index: str, query: str, k: int) -> List[Dict[str, Any]]:
    body = {"query": {"match": {"text": {"query": query}}}, "size": k}
    hits = es.search(index=index, body=body)["hits"]["hits"]
    out = []
    for h in hits:
        item = {"id": h["_id"], "score": h["_score"], **h["_source"]}
        item["source"] = "bm25"
        out.append(item)
    return out

def _dense(es: Elasticsearch, index: str, query: str, k: int) -> List[Dict[str, Any]]:
    vec = _EMB.encode(query, normalize_embeddings=True).tolist()
    body = {
        "knn": {
            "field": "embedding",
            "query_vector": vec,
            "k": k,
            "num_candidates": max(50, k * 10),
        },
        "_source": True,
    }
    hits = es.search(index=index, body=body)["hits"]["hits"]
    out = []
    for h in hits:
        item = {"id": h["_id"], "score": h["_score"], **h["_source"]}
        item["source"] = "dense"
        out.append(item)
    return out

def _elser(es: Elasticsearch, index: str, query: str, k: int) -> List[Dict[str, Any]]:
    """
    ELSER (sparse) via text_expansion over 'ml.tokens'.
    Works with either inference endpoint or model id.
    """
    try:
        te_spec: Dict[str, Any] = {"ml.tokens": {"model_text": query}}
        if ELSER_INFERENCE_ID:
            te_spec["ml.tokens"]["inference_id"] = ELSER_INFERENCE_ID
        else:
            te_spec["ml.tokens"]["model_id"] = ELSER_MODEL_ID

        body = {"query": {"text_expansion": te_spec}, "size": k}
        hits = es.search(index=index, body=body)["hits"]["hits"]
        out = []
        for h in hits:
            item = {"id": h["_id"], "score": h["_score"], **h["_source"]}
            item["source"] = "elser"
            out.append(item)
        return out
    except Exception:
        return []

def _rrf(buckets: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    scores: dict[tuple, float] = {}
    keep: dict[tuple, Dict[str, Any]] = {}

    for bucket in buckets:
        for rank, item in enumerate(bucket, start=1):
            key = (item.get("filename"), item.get("chunk_id"))
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in keep:
                keep[key] = item

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    out = []
    for key, s in ranked:
        item = keep[key].copy()
        item["rrf"] = s
        out.append(item)
    return out

def retrieve_docs(
    es: Elasticsearch,
    index: str,
    question: str,
    mode: str = "hybrid",
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    bm = _bm25(es, index, question, top_k)
    if mode == "elser":
        el = _elser(es, index, question, top_k)
        fused = _rrf([bm, el])
    else:
        de = _dense(es, index, question, top_k)
        el = _elser(es, index, question, top_k)
        # Put ELSER first so at least one fused item retains source='elser'
        fused = _rrf([el, de, bm])
    return fused[:top_k]
