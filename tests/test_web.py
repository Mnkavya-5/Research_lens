import pytest
import requests

from src import pipeline
from src.web import fetch, search

ARTICLE = "<html><head><title>RAG explained</title></head><body><article><h1>RAG explained</h1><p>" + \
          ("Retrieval augmented generation retrieves documents before answering questions. " * 40) + \
          "</p></article></body></html>"


class Resp:
    def __init__(self, text="", status=200, ctype="text/html", data=None):
        self.text, self.status_code, self.headers, self._data = text, status, {"Content-Type": ctype}, data

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


@pytest.fixture(autouse=True)
def robots_allow_all(monkeypatch):
    fetch._robots.cache_clear()
    monkeypatch.setattr(fetch, "allowed_by_robots", lambda url: True)


def test_fetch_extracts_article(monkeypatch):
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: Resp(ARTICLE))
    page = fetch.fetch_page("https://example.com/rag")
    assert page["title"] == "RAG explained" and "retrieves documents" in page["text"]


def test_robots_disallow_is_respected(monkeypatch):
    monkeypatch.setattr(fetch, "allowed_by_robots", lambda url: False)
    with pytest.raises(fetch.FetchError, match="robots.txt"):
        fetch.fetch_page("https://example.com/private")


def test_robots_parser_reads_rules(monkeypatch):
    fetch._robots.cache_clear()
    monkeypatch.undo()  # use the real allowed_by_robots
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: Resp("User-agent: *\nDisallow: /private/"))
    assert fetch.allowed_by_robots("https://example.org/blog/post")
    assert not fetch.allowed_by_robots("https://example.org/private/report")


@pytest.mark.parametrize("resp, message", [
    (Resp(status=403), "login or subscription"),
    (Resp("<html><body><p>Subscribe to read this article.</p></body></html>"), "too little readable text"),
    (Resp(ARTICLE, ctype="application/pdf"), "not a web page"),
])
def test_paywalls_and_non_pages_are_skipped(monkeypatch, resp, message):
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: resp)
    with pytest.raises(fetch.FetchError, match=message):
        fetch.fetch_page("https://example.com/x")


def test_non_web_link_rejected():
    with pytest.raises(fetch.FetchError, match="not a web link"):
        fetch.fetch_page("file:///etc/passwd")


def test_chunk_page_labels_web_sources():
    chunks = fetch.chunk_page({"title": "T", "url": "https://example.com/a", "text": "word " * 300, "published": ""})
    assert chunks[0]["kind"] == "web" and chunks[0]["page"] == 1 and chunks[0]["chunk_id"].startswith("web_")


def test_search_uses_tavily_when_key_set(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    calls = []
    def fake_post(url, headers, json, timeout):
        calls.append((url, headers["Authorization"], json["query"]))
        return Resp(data={"results": [{"title": "A", "url": "https://a.com", "content": "snippet"}]})
    monkeypatch.setattr(search.requests, "post", fake_post)
    results, errors = search.search_web("agentic rag")
    assert calls == [(search.TAVILY_URL, "Bearer tvly-test", "agentic rag")]
    assert results[0]["found_via"] == "tavily" and errors == []


def test_search_falls_back_to_free_sources(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    def fake_get(url, params, headers, timeout):
        if url == search.WIKIPEDIA_URL:
            return Resp(data={"query": {"search": [{"title": "Retrieval-augmented generation"}]}})
        raise requests.ConnectionError("hn down")
    monkeypatch.setattr(search.requests, "get", fake_get)
    results, errors = search.search_web("rag")
    assert results[0]["url"] == "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
    assert len(errors) == 1 and "hackernews" in errors[0]


def test_index_web_pages_keeps_going_after_failures(monkeypatch):
    def fake_fetch(url):
        if "bad" in url:
            raise fetch.FetchError(f"{url} requires a login or subscription (403); skipped")
        return {"title": "Good", "url": url, "text": "word " * 300, "published": ""}
    stored = []
    monkeypatch.setattr(pipeline, "fetch_page", fake_fetch)
    monkeypatch.setattr(pipeline, "add_chunks", lambda col, chunks: stored.extend(chunks))
    pages, n, errors = pipeline.index_web_pages([{"url": "https://good.com"}, {"url": "https://bad.com"}], None)
    assert [p["title"] for p in pages] == ["Good"] and n == len(stored) > 0
    assert len(errors) == 1 and "subscription" in errors[0]
