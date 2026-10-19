import hashlib
from functools import lru_cache
from urllib import robotparser
from urllib.parse import urlparse

import requests
import trafilatura

from src.processing.chunking import split_words
from src.web.search import HEADERS

MIN_WORDS = 150  # shorter pages are usually paywall teasers, cookie walls or error pages
MAX_CHUNKS = 40  # ~5,000 words; long pages (e.g. big Wikipedia articles) would flood the collection


class FetchError(Exception):
    """A web page we may not or could not read."""


@lru_cache(maxsize=256)
def _robots(root: str) -> robotparser.RobotFileParser:
    parser = robotparser.RobotFileParser()
    try:
        response = requests.get(root + "/robots.txt", headers=HEADERS, timeout=10)
        parser.parse(response.text.splitlines() if response.status_code == 200 else [])
    except requests.RequestException:
        parser.parse([])  # robots.txt unreachable: no rules published
    return parser


def allowed_by_robots(url: str) -> bool:
    parts = urlparse(url)
    return _robots(f"{parts.scheme}://{parts.netloc}").can_fetch(HEADERS["User-Agent"], url)


def fetch_page(url: str) -> dict:
    if urlparse(url).scheme not in ("http", "https"):
        raise FetchError(f"not a web link: {url}")
    if not allowed_by_robots(url):
        raise FetchError(f"the site's robots.txt does not allow automated reading of {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
    except requests.RequestException as e:
        raise FetchError(f"could not download {url}: {e}") from e
    if response.status_code in (401, 402, 403):
        raise FetchError(f"{url} requires a login or subscription ({response.status_code}); skipped")
    if response.status_code >= 400:
        raise FetchError(f"{url} returned HTTP {response.status_code}")
    if "html" not in response.headers.get("Content-Type", "html"):
        raise FetchError(f"{url} is not a web page ({response.headers.get('Content-Type')})")

    text = trafilatura.extract(response.text, include_comments=False, include_tables=True) or ""
    if len(text.split()) < MIN_WORDS:
        raise FetchError(f"{url}: too little readable text (paywall, login wall or empty page)")
    meta = trafilatura.extract_metadata(response.text)
    title = (meta.title if meta and meta.title else None) or url
    date = (meta.date if meta and meta.date else "") or ""
    return {"title": title, "url": url, "text": text, "published": date}


def chunk_page(page: dict) -> list[dict]:
    page_id = "web_" + hashlib.sha1(page["url"].encode()).hexdigest()[:12]
    return [{
        "chunk_id": f"{page_id}_c{n}",
        "text": text,
        "page": n + 1,          # for web pages this is the section number, not a printed page
        "title": page["title"],
        "authors": "",
        "published": page.get("published", ""),
        "url": page["url"],
        "pdf_url": "",
        "kind": "web",
    } for n, text in enumerate(split_words(page["text"])[:MAX_CHUNKS])]
