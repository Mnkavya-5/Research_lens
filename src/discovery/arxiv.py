import feedparser
import requests

ARXIV_URL ="https://export.arxiv.org/api/query"


def _find_pdf(entry) -> str | None:
    for link in entry.links:
        if link.get("title")=="pdf":
            return link.href
    return None 


def search_arxiv(query: str, max_results: int = 10) -> list[dict]:
    params = {"search_query": query, "max_results":max_results}
    response = requests.get(ARXIV_URL,params=params, timeout=30)
    response.raise_for_status()
    feed = feedparser.parse(response.text)

    papers =[]
    for entry in feed.entries:
        papers.append({
            "title": entry.title,
            "abstract":entry.summary,
            "authors":[a.name for a in entry.authors],
            "published":entry.published,
            "url": entry.link,
            "pdf_url":_find_pdf(entry),
            "source": "arxiv",

        })
    return papers 