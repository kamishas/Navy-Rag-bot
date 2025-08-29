from __future__ import annotations

import os
import re
import time
from typing import List, Dict, Tuple, Optional

import requests

from .ingest import read_pdf_text, chunk_text, embed
from elasticsearch.helpers import bulk

# ----------------------------
# Optional Google Drive client
# ----------------------------
def _get_drive_service():
    """
    Returns a Google Drive v3 client using a service account.
    Requires env GOOGLE_SERVICE_ACCOUNT_FILE pointing to JSON credentials.
    """
    cred_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "").strip()
    if not cred_path or not os.path.isfile(cred_path):
        return None
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        scopes = ["https://www.googleapis.com/auth/drive.readonly"]
        creds = service_account.Credentials.from_service_account_file(cred_path, scopes=scopes)
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception:
        return None


def _get_authorized_session():
    """
    Authorized requests.Session using service account (for streaming downloads).
    """
    cred_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "").strip()
    if not cred_path or not os.path.isfile(cred_path):
        return None
    try:
        from google.oauth2 import service_account
        from google.auth.transport.requests import AuthorizedSession
        scopes = ["https://www.googleapis.com/auth/drive.readonly"]
        creds = service_account.Credentials.from_service_account_file(cred_path, scopes=scopes)
        return AuthorizedSession(creds)
    except Exception:
        return None


# ----------------------------
# Drive URL helpers
# ----------------------------
_FILE_PATTERNS = [
    r"drive\.google\.com/file/d/([a-zA-Z0-9_-]{20,})",
    r"drive\.google\.com/open\?id=([a-zA-Z0-9_-]{20,})",
    r"drive\.google\.com/uc\?export=download&id=([a-zA-Z0-9_-]{20,})",
]
_FOLDER_PATTERNS = [
    r"drive\.google\.com/drive/folders/([a-zA-Z0-9_-]{20,})",
]


def _extract_file_id(url: str) -> Optional[str]:
    for pat in _FILE_PATTERNS:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def _extract_folder_id(url: str) -> Optional[str]:
    for pat in _FOLDER_PATTERNS:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def _direct_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"


# ----------------------------
# Utilities
# ----------------------------
def _looks_like_pdf(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            head = f.read(5)
        return head == b"%PDF-"
    except Exception:
        return False


def _index_docs(es, index: str, docs: List[Dict], pipeline: Optional[str] = None) -> int:
    """
    Bulk index with optional ingest pipeline (so we don't depend on es_utils.index_docs signature).
    """
    if not docs:
        return 0
    actions = [{"_index": index, "_source": d} for d in docs]
    if pipeline:
        bulk(es, actions, pipeline=pipeline)
    else:
        bulk(es, actions)
    return len(docs)


# ----------------------------
# Public download (verify PDF; fallback if not)
# ----------------------------
def _download_public_pdf(file_id: str, dest_dir: str = "data/tmp", timeout: int = 90, max_retries: int = 3) -> Tuple[str, str]:
    """
    Download a publicly shared PDF by file ID, handling Google's confirm token when needed.
    Returns (local_path, filename). Raises to allow fallback to tokened API stream.
    """
    os.makedirs(dest_dir, exist_ok=True)
    url = _direct_download_url(file_id)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            with requests.Session() as s:
                r = s.get(url, stream=True, timeout=timeout)
                r.raise_for_status()

                # Handle interstitial confirm
                disp = r.headers.get("content-disposition", "") or ""
                if "filename" not in disp.lower():
                    for k, v in r.cookies.items():
                        if k.startswith("download_warning"):
                            r = s.get(url + f"&confirm={v}", stream=True, timeout=timeout)
                            r.raise_for_status()
                            disp = r.headers.get("content-disposition", "") or ""
                            break

                # Decide filename
                name_match = re.search(r'filename="([^"]+)"', disp) or re.search(r"filename\*=UTF-8''([^;]+)", disp)
                filename = name_match.group(1) if name_match else f"{file_id}.pdf"
                if not filename.lower().endswith(".pdf"):
                    filename += ".pdf"
                local_path = os.path.join(dest_dir, filename)

                # Stream to disk
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(1 << 14):
                        if chunk:
                            f.write(chunk)

                # Validate: content-type or magic bytes
                ctype = (r.headers.get("content-type", "") or "").lower()
                if "pdf" not in ctype and not _looks_like_pdf(local_path):
                    # Not a real PDF → remove and raise for fallback
                    try:
                        os.remove(local_path)
                    except Exception:
                        pass
                    raise RuntimeError(f"Public download returned non-PDF content-type: {ctype}")

                # Also validate magic bytes just in case
                if not _looks_like_pdf(local_path):
                    try:
                        os.remove(local_path)
                    except Exception:
                        pass
                    raise RuntimeError("Public download saved file that does not start with %PDF-.")

                return local_path, filename

        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(1.2 * attempt)
            else:
                raise

    if last_err:
        raise last_err
    raise RuntimeError("Unknown public download error")


# ----------------------------
# Tokened streaming download (stable on Windows)
# ----------------------------
def _api_stream_download(file_id: str, dest_path: str, chunk_size: int = 1024 * 1024, max_retries: int = 4, timeout: int = 120):
    """
    Streams the file via Google Drive API using AuthorizedSession.
    This avoids 'Stream has ended unexpectedly' and supports private files the service account can view.
    """
    sess = _get_authorized_session()
    if not sess:
        raise RuntimeError("No AuthorizedSession (missing GOOGLE_SERVICE_ACCOUNT_FILE).")

    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            with sess.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
            # Validate PDF signature
            if not _looks_like_pdf(dest_path):
                try:
                    os.remove(dest_path)
                except Exception:
                    pass
                raise RuntimeError("API stream returned non-PDF content.")
            return
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(1.5 * attempt)


# ----------------------------
# Chunk → metadata helpers
# ----------------------------
_HEADING_PAT = re.compile(r"(?:\bRule\s*\d+\b[^\n]*|\bOvertaking\b)", re.IGNORECASE)
_SECTION_PAT = re.compile(r"(?:\bINTERNATIONAL\b|\bINLAND\b)", re.IGNORECASE)
_PART_SEC_PAT = re.compile(r"(?:Part\s+[A-Z]\b[^\n]*|Section\s+[IVX]+\b[^\n]*)", re.IGNORECASE)


def _guess_labels(text: str):
    heading = None
    section = None
    part_section = None

    m = _HEADING_PAT.search(text or "")
    if m:
        heading = m.group(0).strip()

    m = _SECTION_PAT.search(text or "")
    if m:
        section = m.group(0).strip()

    m = _PART_SEC_PAT.search(text or "")
    if m:
        part_section = m.group(0).strip()

    return heading, section, part_section


def _estimate_page(chunk: str, pages: List[str]) -> Optional[int]:
    prefix = (chunk or "").strip()[:32]
    if not prefix:
        return None
    for i, ptxt in enumerate(pages, start=1):
        if prefix in ptxt:
            return i
    return None


def _docs_from_local_pdf(local_path: str, display_name: str, source_url: str) -> List[Dict]:
    # Try per-page extraction for page estimation; swallow PDF errors gracefully
    pages_text: List[str] = []
    try:
        from pypdf import PdfReader
        reader = PdfReader(local_path)
        for p in reader.pages:
            try:
                pages_text.append(p.extract_text() or "")
            except Exception:
                pages_text.append("")
    except Exception:
        pages_text = []

    # Full-text (swallow errors → skip file)
    try:
        text = read_pdf_text(local_path)
    except Exception:
        return []

    if not text:
        return []

    parts = chunk_text(text)
    embs = embed(parts)

    docs: List[Dict] = []
    for i, (chunk, vec) in enumerate(zip(parts, embs)):
        heading, section, part_section = _guess_labels(chunk)
        page = _estimate_page(chunk, pages_text) if pages_text else None
        docs.append({
            "text": chunk,
            "filename": display_name,
            "url": source_url,
            "chunk_id": f"{display_name}__{i:04d}",
            "embedding": vec,
            "page": page,
            "heading": heading,
            "section": section,
            "part_section": part_section,
            # ml.tokens added by ingest pipeline/backfill
        })
    return docs


# ----------------------------
# Public file URL ingest
# ----------------------------
def _ingest_public_file(es, index: str, file_url: str, pipeline: Optional[str] = None) -> int:
    file_id = _extract_file_id(file_url)
    if not file_id:
        raise ValueError("URL is not a recognized Drive file link.")
    # Try public, then authorized stream
    try:
        local_path, disp_name = _download_public_pdf(file_id)
    except Exception:
        os.makedirs("data/tmp", exist_ok=True)
        disp_name = f"{file_id}.pdf"
        local_path = os.path.join("data/tmp", disp_name)
        _api_stream_download(file_id, local_path)

    docs = _docs_from_local_pdf(local_path, disp_name, file_url)
    return _index_docs(es, index, docs, pipeline=pipeline)


# ----------------------------
# Folder ingest (list via API → download each)
# ----------------------------
def _ingest_folder(es, index: str, folder_url: str, pipeline: Optional[str] = None, limit: Optional[int] = None) -> int:
    folder_id = _extract_folder_id(folder_url)
    if not folder_id:
        raise ValueError("URL is not a recognized Drive folder link.")

    svc = _get_drive_service()
    if not svc:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_FILE not set or missing; cannot crawl a folder.")

    # List PDFs only
    q = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    files: List[Dict] = []
    page_token = None
    while True:
        resp = svc.files().list(
            q=q,
            fields="nextPageToken, files(id,name,webViewLink)",
            pageToken=page_token,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    if limit:
        files = files[: int(limit)]

    os.makedirs("data/tmp", exist_ok=True)

    total = 0
    skipped = 0
    for f in files:
        fid = f["id"]
        name = f.get("name", f"{fid}.pdf")
        web = f.get("webViewLink", folder_url)
        local_path = os.path.join("data/tmp", name)

        # 1) Public direct download
        used_api_stream = False
        try:
            _download_public_pdf(fid, dest_dir="data/tmp")
        except Exception:
            # 2) Authorized stream fallback
            try:
                _api_stream_download(fid, local_path)
                used_api_stream = True
            except Exception:
                skipped += 1
                continue

        # If public path saved with a different filename, prefer the stable 'name'
        if not used_api_stream and not os.path.exists(local_path):
            # The public helper may have saved as <served-filename>.pdf; find it and rename
            candidate = os.path.join("data/tmp", f"{fid}.pdf")
            if os.path.exists(candidate):
                try:
                    os.replace(candidate, local_path)
                except Exception:
                    local_path = candidate  # fall back

        if not _looks_like_pdf(local_path):
            # As a final guard, skip non-PDFs (e.g., HTML error pages)
            try:
                os.remove(local_path)
            except Exception:
                pass
            skipped += 1
            continue

        docs = _docs_from_local_pdf(local_path, name, web)
        total += _index_docs(es, index, docs, pipeline=pipeline)

    if total == 0:
        raise RuntimeError(f"No valid PDFs ingested (skipped={skipped}). Check permissions and file types.")
    return total


# ----------------------------
# Public API
# ----------------------------
def ingest_drive_url(es, index: str, url: str, pipeline: Optional[str] = None, limit: Optional[int] = None) -> int:
    """
    Ingest a Google Drive file OR folder URL.
    - File: tries public download, then authorized stream
    - Folder: lists via Drive API; for each file tries public → tokened stream; skips bad files
    """
    url = (url or "").strip()
    if not url:
        raise ValueError("Empty URL.")

    if _extract_file_id(url):
        return _ingest_public_file(es, index, url, pipeline=pipeline)

    if _extract_folder_id(url):
        return _ingest_folder(es, index, url, pipeline=pipeline, limit=limit)

    raise ValueError("URL is neither a recognized Drive file nor folder link.")
