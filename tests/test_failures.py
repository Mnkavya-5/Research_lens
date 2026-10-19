"""Failure cases that span several modules (Milestone 8 checklist)."""
import pymupdf
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import pipeline
from src.rag import agent
from src.rag import answer as rag
from src.retrieval import embeddings, store
from tests.test_rag import fake_embed


def _pdf(path, text):
    doc = pymupdf.open()
    page = doc.new_page()
    if text:
        page.insert_textbox(pymupdf.Rect(72, 72, 540, 770), text)
    doc.save(path)
    doc.close()


def test_pipeline_skips_scanned_pdf_and_keeps_going(monkeypatch, tmp_path):
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    good, scanned = tmp_path / "good.pdf", tmp_path / "scan.pdf"
    _pdf(good, "Agentic retrieval augmented generation is evaluated on faithfulness and latency. " * 6)
    _pdf(scanned, "")
    papers = [{"title": "Good", "pdf_url": "https://arxiv.org/pdf/good", "pdf_path": str(good)},
              {"title": "Scanned", "pdf_url": "https://arxiv.org/pdf/scan", "pdf_path": str(scanned)}]
    monkeypatch.setattr(pipeline, "download_all", lambda ps: (ps, []))  # pretend both downloaded
    col = store.get_collection(str(tmp_path / "chroma"))

    n_chunks, errors = pipeline.index_papers(papers, col)

    assert n_chunks > 0 and col.count() == n_chunks
    assert len(errors) == 1 and "scanned" in errors[0]


def test_ingest_reports_errors_from_every_stage(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "discover", lambda topic: (
        [{"title": "No PDF", "pdf_url": None}, {"title": "Has PDF", "pdf_url": "u"}], ["crossref failed: timeout"]))
    monkeypatch.setattr(pipeline, "index_papers", lambda papers, col: (0, ["download failed for 'Has PDF'"]))
    result = pipeline.ingest("rag", collection=object(), max_pdfs=5)
    assert result["errors"] == ["crossref failed: timeout", "download failed for 'Has PDF'"]
    assert len(result["papers"]) == 2  # metadata-only papers are still reported


def test_empty_collection_refuses_without_calling_llm(monkeypatch, tmp_path):
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    def must_not_call(*a, **k):
        raise AssertionError("LLM should not be called with no sources")
    monkeypatch.setattr(rag, "ask_ollama", must_not_call)
    result = rag.answer_question(store.get_collection(str(tmp_path / "chroma")), "anything?")
    assert result["answer"] == rag.NOT_FOUND and result["warning"] == "no papers indexed"


def test_app_shows_error_when_ollama_is_down(monkeypatch):
    class OneChunk:
        def count(self):
            return 1
    def ollama_down(*a, **k):
        raise requests.ConnectionError("Connection refused")
    st.cache_resource.clear()
    monkeypatch.setattr(store, "get_collection", lambda *a, **k: OneChunk())
    monkeypatch.setattr(agent, "research_answer", ollama_down)

    at = AppTest.from_file("../app.py", default_timeout=60).run()
    at.chat_input[0].set_value("What datasets are used?").run()

    assert not at.exception  # no crash page
    assert "ollama serve" in at.error[0].value
    st.cache_resource.clear()
