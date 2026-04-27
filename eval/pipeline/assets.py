from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests


def resolve_paper_download_url(url: str) -> str:
    """Normalize benchmark paper URLs to a direct downloadable PDF URL."""
    parsed = urlparse(url)
    if parsed.netloc == "openreview.net" and parsed.path == "/forum":
        paper_id = parse_qs(parsed.query).get("id", [None])[0]
        if not paper_id:
            raise ValueError(f"OpenReview URL missing paper id: {url}")
        return f"https://openreview.net/pdf?id={paper_id}"
    return url


def download_file(url: str, output_path: str | Path) -> Path:
    """Download one URL to a stable local path, skipping work when it already exists."""
    local_candidate = Path(url).expanduser()
    if local_candidate.exists():
        return local_candidate.resolve()
    parsed = urlparse(url)
    if not parsed.scheme:
        local_path = Path(url).expanduser().resolve()
        if not local_path.exists():
            raise FileNotFoundError(f"Local asset path not found: {local_path}")
        return local_path
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    response = requests.get(url, timeout=120, allow_redirects=True)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def hashed_filename(task_id: str, url: str, suffix: str) -> str:
    """Build a stable local filename so reruns reuse the same downloaded artifact."""
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{task_id}_{digest}{suffix}"
