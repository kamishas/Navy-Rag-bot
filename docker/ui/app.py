# docker/ui/app.py
import os
import requests
import streamlit as st
from html import escape

DEFAULT_API = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG: Elastic + Open LLM", page_icon="🔎", layout="wide")

# ---------- small CSS polish ----------
st.markdown(
    """
    <style>
      :root {
        --card-bg: rgba(255,255,255,0.03);
        --card-border: rgba(120,120,120,0.25);
        --muted: rgba(140,140,160,0.95);
        --badge-bg: rgba(120,120,140,0.18);
        --badge-border: rgba(120,120,160,0.35);
      }
      .rag-title { font-weight: 800; font-size: 2.2rem; letter-spacing: .2px; }
      .pill-row { display: flex; gap: .5rem; flex-wrap: wrap; margin: .25rem 0 .5rem 0; }
      .badge { display:inline-block; padding:.22rem .5rem; border-radius:999px;
               background:var(--badge-bg); border:1px solid var(--badge-border); font-size:.78rem; }
      .card { border:1px solid var(--card-border); background:var(--card-bg);
              border-radius:14px; padding:.9rem 1rem; margin:.5rem 0 1rem 0; }
      .card h4 { margin:0 0 .35rem 0; font-weight:700; }
      .muted { color:var(--muted); font-size:.9rem; }
      .link { text-decoration:none; }
      .answer-box { border-left:4px solid #7c83ff; padding-left:1rem; margin-top:.25rem; }
      .examples { display:flex; gap:.4rem; flex-wrap:wrap; }
      .chip { padding:.35rem .65rem; border-radius:999px; border:1px solid var(--badge-border);
              background:var(--badge-bg); cursor:pointer; font-size:.86rem; }
      .chip:hover { filter:brightness(1.1); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="rag-title">🔎 RAG (Elastic + Open LLM)</div>', unsafe_allow_html=True)
st.caption("Ask questions over your Google Drive PDFs with hybrid retrieval (BM25 + dense + ELSER).")

# ---------- helpers ----------
if "api_url" not in st.session_state:
    st.session_state.api_url = os.getenv("API_URL", DEFAULT_API)
if "query" not in st.session_state:
    st.session_state.query = ""

def post_json(path: str, payload: dict, timeout: int = 120):
    url = f"{st.session_state.api_url}{path}"
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()

def get_json(path: str, timeout: int = 15):
    url = f"{st.session_state.api_url}{path}"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def badge(text: str) -> str:
    return f'<span class="badge">{escape(str(text))}</span>'

def card_header_link(title: str, url: str) -> str:
    safe_title = escape(title or "document")
    safe_url = escape(url or "#")
    return f'<h4><a class="link" href="{safe_url}" target="_blank">{safe_title}</a></h4>'

# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    api_in = st.text_input("API base URL", value=st.session_state.api_url, help="Your FastAPI base URL")
    if api_in and api_in != st.session_state.api_url:
        st.session_state.api_url = api_in

    colA, colB = st.columns([1,1])
    with colA:
        mode = st.radio("Retrieval mode", ["hybrid", "elser"], index=0, help="Use only ELSER or fuse BM25+dense+ELSER")
    with colB:
        top_k = st.slider("Top K", min_value=3, max_value=20, value=5, step=1)

    debug = st.toggle("Show debug chunks", value=False)

    if st.button("Test connection", use_container_width=True):
        try:
            h = get_json("/healthz")
            st.success(f"Connected ✅  {h}")
        except Exception as e:
            st.error(f"Connection failed: {e}")
    st.caption(f"Backend → {st.session_state.api_url}")

# ---------- main input (no form) ----------
st.session_state.query = st.text_input(
    "Ask a question",
    value=st.session_state.query,
    placeholder="e.g., List the factors used to determine safe speed under Rule 6.",
)

colAsk, _ = st.columns([1, 3])
with colAsk:
    submit = st.button("Ask", use_container_width=True)

# example chips (outside any form)
exs = [
    "Summarize Rule 13 (Overtaking) briefly.",
    "What sound signals are prescribed under Rule 34?",
    "Explain Rule 10 (Traffic Separation) in 3 bullets.",
    "What are the responsibilities between vessels (Rule 18)?",
]
st.markdown('<div class="examples">', unsafe_allow_html=True)
chip_cols = st.columns(len(exs))
for i, ex in enumerate(exs):
    if chip_cols[i].button(ex, key=f"ex_{i}", help="Use this example"):
        st.session_state.query = ex
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

# ---------- results ----------
q = st.session_state.query
if submit and q and q.strip():
    try:
        if debug:
            with st.spinner("Retrieving (debug)…"):
                data = post_json("/query_debug", {"question": q, "top_k": top_k, "mode": mode})

            st.subheader("Debug hits")
            hits = data.get("hits", [])
            if not hits:
                st.info("No hits.")
            else:
                for i, h in enumerate(hits, start=1):
                    title = h.get("filename", "(no name)")
                    url = h.get("url") or "#"
                    rrf = h.get("rrf")
                    score = h.get("score")
                    source = h.get("source", None)

                    meta_badges = []
                    if source: meta_badges.append(badge(source))
                    if rrf is not None: meta_badges.append(badge(f"rrf {round(rrf, 5)}"))
                    if score is not None: meta_badges.append(badge(f"score {round(score, 3)}"))
                    if h.get("page"): meta_badges.append(badge(f"page {h['page']}"))
                    if h.get("heading"): meta_badges.append(badge(f"heading: {h['heading']}"))
                    if h.get("section"): meta_badges.append(badge(f"section: {h['section']}"))
                    if h.get("part_section"): meta_badges.append(badge(f"part: {h['part_section']}"))
                    if h.get("chunk_id"): meta_badges.append(badge(f"id: {h['chunk_id']}"))

                    st.markdown(
                        '<div class="card">' + card_header_link(title, url) +
                        f'<div class="pill-row">{" ".join(meta_badges)}</div>' +
                        '</div>', unsafe_allow_html=True
                    )
                    st.code(h.get("snippet", "") or "", language="text")

        else:
            with st.spinner("Answering…"):
                data = post_json("/query", {"question": q, "top_k": top_k, "mode": mode})

            st.subheader("Answer")
            st.markdown(
                f'<div class="card answer-box">{escape(data.get("answer","(no answer)"))}</div>',
                unsafe_allow_html=True
            )

            st.subheader("Citations")
            cites = data.get("citations", [])
            if not cites:
                st.info("No citations.")
            else:
                for c in cites:
                    title = c.get("title", "document")
                    url = c.get("link", "#")
                    meta_badges = []
                    if c.get("page"): meta_badges.append(badge(f"page {c['page']}"))
                    if c.get("heading"): meta_badges.append(badge(f"heading: {c['heading']}"))
                    if c.get("section"): meta_badges.append(badge(f"section: {c['section']}"))
                    if c.get("part_section"): meta_badges.append(badge(f"part: {c['part_section']}"))
                    if c.get("chunk_id"): meta_badges.append(badge(f"c id: {c['chunk_id']}"))

                    st.markdown(
                        '<div class="card">' + card_header_link(title, url) +
                        f'<div class="pill-row">{" ".join(meta_badges)}</div>' +
                        '</div>', unsafe_allow_html=True
                    )
                    st.code((c.get("snippet","") or "")[:300], language="text")

    except requests.HTTPError as e:
        st.error(f"API error: {e.response.text if e.response is not None else e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
