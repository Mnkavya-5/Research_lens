import os

import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env into environment variables
CROSSREF_URL = "https://api.crossref.org/works"
SKIP_TYPES = {"component", "peer-review"}  # single figures/tables and reviews, not papers


def _format_authors(raw: list[dict]) -> list[str]:
    names = []
    for a in raw:
        full = f"{a.get('given', '')} {a.get('family', '')}".strip()
        if full:
            names.append(full)
    return names


def _format_date(item: dict) -> str | None:
    parts = item.get("published", {}).get("date-parts", [[None]])[0]
    return "-".join(str(p) for p in parts) if parts[0] else None


def search_crossref(query: str, max_results: int = 10) -> list[dict]:
    params = {
        "query": query,
        "rows": max_results,
        "mailto": os.getenv("CONTACT_EMAIL"),
    }
    response = requests.get(CROSSREF_URL, params=params, timeout=30)
    response.raise_for_status()
    items = response.json()["message"]["items"]

    papers = []
    for item in items:
        if item.get("type") in SKIP_TYPES:
            continue
        papers.append({
            "title": (item.get("title") or ["Untitled"])[0],
            "abstract": item.get("abstract"),
            "authors": _format_authors(item.get("author", [])),
            "published": _format_date(item),
            "url": item.get("URL"),
            "pdf_url": None,  # never use publisher/similarity-checking links
            "source": "crossref",
        })
    return papers
