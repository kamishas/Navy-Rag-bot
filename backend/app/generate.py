from __future__ import annotations

import os
from typing import List, Dict

# Respect OLLAMA_BASE_URL if set
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
if OLLAMA_BASE_URL:
    os.environ["OLLAMA_HOST"] = OLLAMA_BASE_URL

try:
    from ollama import chat  # type: ignore
except Exception:
    chat = None  # LLM optional

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

SYSTEM = (
    "You are a helpful assistant that answers ONLY using the provided context.\n"
    "If the answer is not present in the context, say: I don't know.\n"
    "Keep answers concise. Include no speculation."
)

def _format_prompt(question: str, docs: List[Dict]) -> str:
    ctx_blocks = []
    for i, d in enumerate(docs, start=1):
        title = d.get("filename", f"doc-{i}")
        url = d.get("url", "")
        text = d.get("text", "")
        ctx_blocks.append(f"[{i}] {title} {url}\n{text}\n")
    context = "\n---\n".join(ctx_blocks)
    return f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"

def generate_answer(question: str, docs: List[Dict]) -> str:
    prompt = _format_prompt(question, docs)
    if chat is None:
        joined = " ".join([d.get("text", "") for d in docs])[:900]
        return f"(LLM offline) Based on the retrieved context: {joined}"
    try:
        resp = chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.1},
        )
        return resp["message"]["content"].strip()
    except Exception:
        joined = " ".join([d.get("text", "") for d in docs])[:900]
        return f"(LLM offline) Based on the retrieved context: {joined}"
