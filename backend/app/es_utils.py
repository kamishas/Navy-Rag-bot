from __future__ import annotations

import json
import os
from typing import List, Dict, Optional

from elasticsearch import Elasticsearch

def get_es() -> Elasticsearch:
    url = os.getenv("ELASTIC_URL", "http://localhost:9200")
    return Elasticsearch(hosts=[url])

def _default_body() -> Dict:
    return {
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "filename": {"type": "keyword"},
                "url": {"type": "keyword"},
                "chunk_id": {"type": "keyword"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": 384,
                    "index": True,
                    "similarity": "cosine",
                },
                "ml.tokens": {"type": "sparse_vector"},
                # new metadata
                "page": {"type": "integer"},
                "heading": {"type": "keyword"},
                "section": {"type": "keyword"},
                "part_section": {"type": "keyword"},
            }
        }
    }

def ensure_index(es: Elasticsearch, index: str):
    if es.indices.exists(index=index):
        return
    mapping_path = os.path.join("docker", "elastic-mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            body = json.load(f)
    else:
        body = _default_body()
    # remove OpenSearch-only settings if present
    if "settings" in body and "index" in body["settings"]:
        body["settings"]["index"].pop("knn", None)
    es.indices.create(index=index, **body)

def index_docs(es: Elasticsearch, index: str, docs: List[Dict], pipeline: Optional[str] = None) -> int:
    from elasticsearch.helpers import bulk
    if not docs:
        return 0
    actions = [{"_index": index, "_source": d} for d in docs]
    if pipeline:
        bulk(es, actions, pipeline=pipeline)
    else:
        bulk(es, actions)
    es.indices.refresh(index=index)
    return len(docs)
