import hashlib
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
PDF_DIR = Path("data/pdfs")
DOWNLOAD_DELAY = 3  # seconds between real downloads, same politeness as the arXiv API
HEADERS = {"User-Agent": f"ResearchLens/0.1 (mailto:{os.getenv('CONTACT_EMAIL', 'unknown')})"}


def pdf_filename(pdf_url: str) -> str:
    # "https://arxiv.org/pdf/2511.22858v1" -> "2511.22858v1.pdf"
    last = pdf_url.rstrip("/").split("/")[-1]
    name = re.sub(r"[^\w.\-]", "_", last)
    if not name or name in {".", ".."}:
        name = hashlib.sha1(pdf_url.encode()).hexdigest()[:16]
    return name if name.endswith(".pdf") else name + ".pdf"


def download_pdf(paper: dict, folder: Path = PDF_DIR) -> Path | None:
    url = paper.get("pdf_url")
    if not url:
        return None  # no legal open PDF known for this paper

    path = folder / pdf_filename(url)
    if path.exists() and path.stat().st_size > 0:
        return path  # cached: never download the same file twice

    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError(f"not a PDF (got {response.headers.get('Content-Type')})")

    folder.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(response.content)
    tmp.rename(path)  # only complete files ever get the .pdf name
    return path


def download_all(papers: list[dict], folder: Path = PDF_DIR) -> tuple[list[dict], list[str]]:
    downloaded, errors = [], []
    for paper in papers:
        if not paper.get("pdf_url"):
            continue
        cached = (folder / pdf_filename(paper["pdf_url"])).exists()
        try:
            path = download_pdf(paper, folder)
            downloaded.append({**paper, "pdf_path": str(path)})
        except (requests.RequestException, ValueError, OSError) as e:
            errors.append(f"download failed for '{paper['title'][:60]}': {e}")
        if not cached:
            time.sleep(DOWNLOAD_DELAY)
    return downloaded, errors
