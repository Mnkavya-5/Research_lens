import pytest
import requests

from src.processing import download as dl


class FakeResponse:
    def __init__(self, content, status=200):
        self.content = content
        self.status_code = status
        self.headers = {"Content-Type": "application/pdf" if content.startswith(b"%PDF") else "text/html"}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


PAPER = {"title": "RAG Survey", "pdf_url": "https://arxiv.org/pdf/2511.22858v1"}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)


def test_filename_from_arxiv_url():
    assert dl.pdf_filename("https://arxiv.org/pdf/2511.22858v1") == "2511.22858v1.pdf"


def test_no_pdf_url_returns_none(tmp_path):
    assert dl.download_pdf({"title": "x", "pdf_url": None}, tmp_path) is None


def test_downloads_and_saves(monkeypatch, tmp_path):
    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: FakeResponse(b"%PDF-1.7 data"))
    path = dl.download_pdf(PAPER, tmp_path)
    assert path.read_bytes() == b"%PDF-1.7 data"


def test_cached_file_not_downloaded_again(monkeypatch, tmp_path):
    (tmp_path / "2511.22858v1.pdf").write_bytes(b"%PDF-1.7 old")
    def must_not_call(*a, **k):
        raise AssertionError("should use the cached file")
    monkeypatch.setattr(dl.requests, "get", must_not_call)
    assert dl.download_pdf(PAPER, tmp_path).read_bytes() == b"%PDF-1.7 old"


def test_html_page_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(dl.requests, "get", lambda *a, **k: FakeResponse(b"<html>Please log in</html>"))
    with pytest.raises(ValueError):
        dl.download_pdf(PAPER, tmp_path)
    assert list(tmp_path.iterdir()) == []  # nothing half-saved


def test_download_all_reports_errors_and_continues(monkeypatch, tmp_path):
    good = {"title": "Good", "pdf_url": "https://arxiv.org/pdf/1"}
    bad = {"title": "Bad", "pdf_url": "https://arxiv.org/pdf/2"}
    no_pdf = {"title": "Crossref only", "pdf_url": None}
    def fake_get(url, **k):
        return FakeResponse(b"%PDF ok") if url.endswith("/1") else FakeResponse(b"", status=404)
    monkeypatch.setattr(dl.requests, "get", fake_get)
    downloaded, errors = dl.download_all([good, bad, no_pdf], tmp_path)
    assert [p["title"] for p in downloaded] == ["Good"]
    assert len(errors) == 1 and "Bad" in errors[0]
