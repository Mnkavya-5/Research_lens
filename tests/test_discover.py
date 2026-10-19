import requests

from src.discovery import discover as d


def _paper(title, source):
    return {"title": title, "abstract": None, "pdf_url": None, "source": source}


def _setup(monkeypatch, arxiv, crossref):
    monkeypatch.setattr(d, "generate_queries", lambda topic: [topic])
    monkeypatch.setattr(d, "search_arxiv", arxiv)
    monkeypatch.setattr(d, "search_crossref", crossref)
    monkeypatch.setattr(d.time, "sleep", lambda s: None)  # keep tests fast


def test_to_arxiv_query_drops_stopwords():
    assert d.to_arxiv_query("How do people evaluate RAG?") == "all:evaluate AND all:rag"


def test_results_merged_and_deduped(monkeypatch):
    _setup(monkeypatch,
           lambda q, n: [_paper("RAG Survey", "arxiv")],
           lambda q, n: [_paper("rag survey.", "crossref"), _paper("Other", "crossref")])
    papers, errors = d.discover("rag")
    assert len(papers) == 2
    assert errors == []


def test_one_source_failing_keeps_the_other(monkeypatch):
    def broken(q, n):
        raise requests.ConnectionError("arXiv down")
    _setup(monkeypatch, broken, lambda q, n: [_paper("RAG Survey", "crossref")])
    papers, errors = d.discover("rag")
    assert len(papers) == 1
    assert len(errors) == 1 and "arxiv failed" in errors[0]


def test_retry_succeeds_after_one_failure(monkeypatch):
    calls = []
    def flaky(q, n):
        calls.append(1)
        if len(calls) == 1:
            raise requests.Timeout("slow")
        return [_paper("RAG Survey", "arxiv")]
    _setup(monkeypatch, flaky, lambda q, n: [])
    papers, errors = d.discover("rag")
    assert len(calls) == 2 and len(papers) == 1 and errors == []
