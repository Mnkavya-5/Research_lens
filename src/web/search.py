import os

import requests
from dotenv import load_dotenv

load_dotenv()
TAVILY_URL = "https://api.tavily.com/search"
WIKIPEDIA_URL = "https://en.wikipedia.org/w/api.php"
HN_URL = "https://hn.algolia.com/api/v1/search"
HEADERS = {"User-Agent": f"ResearchLens/0.1 (mailto:{os.getenv('CONTACT_EMAIL', 'unknown')})"}


def search_tavily(query: str, max_results: int, api_key: str) -> list[dict]:
    response = requests.post(TAVILY_URL, headers={"Authorization": f"Bearer {api_key}"},
                             json={"query": query, "max_results": max_results, "search_depth": "basic"},
                             timeout=30)  # basic search = 1 credit (free tier: 1,000 credits/month)
    response.raise_for_status()
    return [{"title": r["title"], "url": r["url"], "snippet": r.get("content", ""), "found_via": "tavily"}
            for r in response.json().get("results", [])]


def search_wikipedia(query: str, max_results: int) -> list[dict]:
    params = {"action": "query", "list": "search", "srsearch": query, "srlimit": max_results, "format": "json"}
    response = requests.get(WIKIPEDIA_URL, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return [{"title": r["title"], "url": "https://en.wikipedia.org/wiki/" + r["title"].replace(" ", "_"),
             "snippet": "", "found_via": "wikipedia"} for r in response.json()["query"]["search"]]


def search_hackernews(query: str, max_results: int) -> list[dict]:
    """Hacker News stories link to blog posts and technical articles; free official API."""
    params = {"query": query, "tags": "story", "hitsPerPage": max_results}
    response = requests.get(HN_URL, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return [{"title": h["title"], "url": h["url"], "snippet": "", "found_via": "hackernews"}
            for h in response.json()["hits"] if h.get("url") and h.get("title")]


def search_web(query: str, max_results: int = 5) -> tuple[list[dict], list[str]]:
    """Tavily when TAVILY_API_KEY is set, otherwise free Wikipedia + Hacker News search.
    Returns (results, errors); one failing source never stops the others."""
    api_key = os.getenv("TAVILY_API_KEY")
    sources = ([("tavily", lambda: search_tavily(query, max_results, api_key))] if api_key else
               [("wikipedia", lambda: search_wikipedia(query, 2)),
                ("hackernews", lambda: search_hackernews(query, max_results))])
    results, errors, seen = [], [], set()
    for name, run in sources:
        try:
            for r in run():
                if r["url"] not in seen:
                    seen.add(r["url"])
                    results.append(r)
        except (requests.RequestException, KeyError, ValueError) as e:
            errors.append(f"{name} search failed: {e}")
    return results[:max_results], errors
