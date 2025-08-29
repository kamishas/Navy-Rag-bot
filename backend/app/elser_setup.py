# backend/app/elser_setup.py
from __future__ import annotations

import os
import time
from typing import Dict, Any

import requests

ES_URL = os.getenv("ELASTIC_URL", "http://localhost:9200").rstrip("/")
DEFAULT_ENDPOINT_ID = os.getenv("ELSER_INFERENCE_ID", "my-elser-endpoint")
DEFAULT_PIPELINE_ID = os.getenv("ELSER_PIPELINE_ID", "elser-v2-mltokens")


def _ok(status: int) -> bool:
    return 200 <= status < 300


def start_trial() -> Dict[str, Any]:
    """Start (or noop) the basic trial so ML/ELSER is allowed."""
    try:
        r = requests.post(f"{ES_URL}/_license/start_trial", params={"acknowledge": "true"}, timeout=30)
        if r.status_code in (200, 202, 400):  # 400 often means trial already started
            return {"ok": True, "status": r.status_code, "body": r.json()}
        return {"ok": False, "status": r.status_code, "body": r.text}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def upsert_elser_endpoint(endpoint_id: str = DEFAULT_ENDPOINT_ID) -> Dict[str, Any]:
    """
    Create the ELSER inference endpoint. If it already exists, that's fine.
    In ES 8.15+, using service "elser" auto-downloads the .elser_model_2.
    """
    body = {
        "service": "elser",
        "service_settings": {
            "num_allocations": 1,
            "num_threads": 1,
        },
    }
    try:
        r = requests.put(f"{ES_URL}/_inference/sparse_embedding/{endpoint_id}", json=body, timeout=60)
        if _ok(r.status_code):
            return {"ok": True, "created": True, "status": r.status_code, "body": r.json()}
        # if it already exists, ES returns 400 with a helpful message
        txt = r.text.lower()
        if "must be unique" in txt or "already exists" in txt:
            return {"ok": True, "created": False, "status": r.status_code, "body": r.text}
        return {"ok": False, "status": r.status_code, "body": r.text}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def wait_endpoint_ready(endpoint_id: str = DEFAULT_ENDPOINT_ID, timeout_s: int = 180) -> Dict[str, Any]:
    """
    Poll the endpoint for a short time; return status snapshot (best effort).
    """
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            r = requests.get(f"{ES_URL}/_inference/sparse_embedding/{endpoint_id}", timeout=15)
            last = {"status": r.status_code, "body": r.json() if _ok(r.status_code) else r.text}
            if _ok(r.status_code):
                # If it's resolvable, proceed (model may still be warming up, that's okay)
                return {"ok": True, **last}
        except Exception as e:
            last = {"error": str(e)}
        time.sleep(2.0)
    return {"ok": False, "last": last}


def put_pipeline(pipeline_id: str = DEFAULT_PIPELINE_ID, endpoint_id: str = DEFAULT_ENDPOINT_ID) -> Dict[str, Any]:
    """
    Create/overwrite an ingest pipeline that writes ELSER tokens into ml.tokens using our endpoint.
    """
    body = {
        "processors": [
            {
                "inference": {
                    "model_id": endpoint_id,  # inference endpoint id
                    "input_output": [
                        {"input_field": "text", "output_field": "ml.tokens"}
                    ],
                }
            }
        ]
    }
    try:
        r = requests.put(f"{ES_URL}/_ingest/pipeline/{pipeline_id}", json=body, timeout=60)
        if _ok(r.status_code):
            return {"ok": True, "status": r.status_code, "body": r.json()}
        return {"ok": False, "status": r.status_code, "body": r.text}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def backfill_tokens(index: str, pipeline_id: str = DEFAULT_PIPELINE_ID) -> Dict[str, Any]:
    """
    Run the pipeline over existing docs to fill ml.tokens.
    """
    try:
        r = requests.post(
            f"{ES_URL}/{index}/_update_by_query",
            params={"pipeline": pipeline_id, "conflicts": "proceed", "refresh": "true"},
            json={"query": {"match_all": {}}},
            timeout=600,
        )
        if _ok(r.status_code):
            return {"ok": True, "status": r.status_code, "body": r.json()}
        return {"ok": False, "status": r.status_code, "body": r.text}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def setup_all(index: str, endpoint_id: str = DEFAULT_ENDPOINT_ID, pipeline_id: str = DEFAULT_PIPELINE_ID,
              do_trial: bool = True, do_backfill: bool = True) -> Dict[str, Any]:
    """
    One-shot setup: trial → endpoint → pipeline → (optional) backfill.
    """
    out: Dict[str, Any] = {}
    if do_trial:
        out["trial"] = start_trial()
    out["endpoint"] = upsert_elser_endpoint(endpoint_id)
    out["endpoint_status"] = wait_endpoint_ready(endpoint_id)
    out["pipeline"] = put_pipeline(pipeline_id, endpoint_id)
    if do_backfill:
        out["backfill"] = backfill_tokens(index, pipeline_id)
    return out
