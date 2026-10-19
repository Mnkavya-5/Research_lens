import re
import time

import requests

from src.discovery.arxiv import search_arxiv
from src.discovery.crossref import search_crossref
from src.discovery.dedupe import dedupe_papers
from src.discovery.query_gen import generate_queries

ARXIV_DELAY = 3  # seconds; arXiv asks for at most one request every 3 s
RETRIES = 3
STOPWORDS = {"a", "an", "the", "of", "for", "in", "on", "and", "or", "to", "with",
             "how", "do", "does", "what", "is", "are", "people", "by", "about", "vs", "versus"}


def to_arxiv_query(query: str) -> str:
    words = [w for w in re.findall(r"\w+", query.lower()) if w not in STOPWORDS]
    return " AND ".join(f"all:{w}" for w in words)


def _with_retry(search, query: str, max_results: int) -> list[dict]:
    for attempt in range(RETRIES):
        try:
            return search(query, max_results)
        except requests.RequestException:
            if attempt == RETRIES - 1:
                raise
            time.sleep(2 ** attempt)  # back off: 1 s, then 2 s
    return []


def discover(topic: str, per_query: int = 10) -> tuple[list[dict], list[str]]:
    papers, errors = [], []
    for i, query in enumerate(generate_queries(topic)):
        if i > 0:
            time.sleep(ARXIV_DELAY)
        sources = [("arxiv", search_arxiv, to_arxiv_query(query)),
                   ("crossref", search_crossref, query)]
        for name, search, q in sources:
            try:
                papers.extend(_with_retry(search, q, per_query))
            except requests.RequestException as e:
                errors.append(f"{name} failed for '{query}': {e}")
    return dedupe_papers(papers), errors
