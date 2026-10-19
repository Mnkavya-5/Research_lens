import pytest
import requests

from src.ranking import baseline, judge
from src.retrieval import embeddings, store
from tests.test_rag import fake_embed


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    monkeypatch.setattr(embeddings, "embed", fake_embed)


def _paper(title, abstract=None, published="2025-01-01", pdf=True):
    return {"title": title, "abstract": abstract, "published": published,
            "pdf_url": f"https://arxiv.org/pdf/{title}" if pdf else None}


def test_paper_year_handles_all_formats():
    assert baseline.paper_year({"published": "2025-11-28T03:28:27Z"}) == 2025
    assert baseline.paper_year({"published": "2026-6"}) == 2026
    assert baseline.paper_year({"published": None}) is None


def test_recency_score():
    assert baseline.recency_score(2026, this_year=2026) == 1.0
    assert baseline.recency_score(2021, this_year=2026) == 0.5
    assert baseline.recency_score(1990, this_year=2026) == 0.0
    assert baseline.recency_score(None) == 0.0


def test_relevant_paper_ranks_first_and_parts_explain_score():
    papers = [_paper("cooking pasta recipes", "boil water add salt"),
              _paper("rag evaluation metrics", "evaluation metrics for rag systems")]
    ranked = baseline.rank_papers("rag evaluation metrics", papers)
    assert ranked[0]["title"] == "rag evaluation metrics"
    parts = ranked[0]["score_parts"]
    expected = sum(baseline.WEIGHTS[k] * v for k, v in parts.items())
    assert ranked[0]["score"] == round(expected, 3)


def test_open_pdf_breaks_tie():
    papers = [_paper("rag evaluation", pdf=False), _paper("rag evaluation", pdf=True)]
    ranked = baseline.rank_papers("rag evaluation", papers)
    assert ranked[0]["pdf_url"] is not None


def test_parse_judgement():
    assert judge.parse_judgement('Sure! {"score": 4, "reason": "Studies RAG metrics."}') == (4, "Studies RAG metrics.")
    assert judge.parse_judgement("I think it is good")[0] is None
    assert judge.parse_judgement('{"score": 9, "reason": "x"}')[0] is None


def test_judge_uses_full_text_when_indexed(monkeypatch, tmp_path):
    col = store.get_collection(str(tmp_path / "chroma"))
    paper = _paper("rag paper", abstract="short abstract")
    store.add_chunks(col, [{"chunk_id": "x_p2_c0", "text": "we evaluate rag faithfulness", "page": 2,
                            "title": "rag paper", "authors": "", "published": "", "url": "", "pdf_url": paper["pdf_url"]}])
    monkeypatch.setattr(judge, "ask_ollama", lambda prompt, **kw: '{"score": 5, "reason": "On topic."}')
    result = judge.judge_papers("rag faithfulness", [paper], col)[0]
    assert result["evidence_type"] == "full text, 1 passages"
    assert result["llm_score"] == 5


def test_judge_falls_back_to_abstract_and_survives_ollama_down(monkeypatch):
    def down(prompt, **kw):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(judge, "ask_ollama", down)
    result = judge.judge_papers("rag", [_paper("rag paper", abstract="about rag", pdf=False)])[0]
    assert result["evidence_type"] == "abstract only"
    assert result["llm_score"] is None
    assert judge.combined_score({**result, "score": 0.7}) == 0.7  # falls back to baseline


def test_combined_score():
    assert judge.combined_score({"score": 0.6, "llm_score": 5}) == 0.8
    assert judge.combined_score({"score": 0.6, "llm_score": 1}) == 0.3


def test_crossref_skips_figure_records(monkeypatch):
    from src.discovery import crossref

    class Resp:
        def raise_for_status(self): pass
        def json(self):
            return {"message": {"items": [
                {"type": "component", "title": ["Table 2: Overall performance metrics"]},
                {"type": "journal-article", "title": ["Agentic RAG Systems"]},
            ]}}
    monkeypatch.setattr(crossref.requests, "get", lambda *a, **k: Resp())
    assert [p["title"] for p in crossref.search_crossref("rag")] == ["Agentic RAG Systems"]


def test_relevance_is_scaled_within_result_set():
    papers = [_paper("rag evaluation metrics"), _paper("rag evaluation"), _paper("cooking pasta")]
    ranked = baseline.rank_papers("rag evaluation metrics", papers, normalize=True)
    rel = [p["score_parts"]["relevance"] for p in ranked]
    assert rel[0] == 1.0 and rel[-1] == 0.0
    assert all("similarity" in p for p in ranked)  # raw value kept for transparency


def test_single_paper_gets_full_relevance():
    assert baseline.rank_papers("rag", [_paper("rag")], normalize=True)[0]["score_parts"]["relevance"] == 1.0


def test_default_uses_raw_similarity():
    ranked = baseline.rank_papers("rag evaluation", [_paper("rag evaluation"), _paper("cooking")])
    assert ranked[0]["score_parts"]["relevance"] == ranked[0]["similarity"] or \
        abs(ranked[0]["score_parts"]["relevance"] - ranked[0]["similarity"]) < 0.001
