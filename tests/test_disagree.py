import json

import pytest

from src.gaps import disagree
from src.retrieval import embeddings, store
from tests.test_rag import fake_embed

A = {"title": "Paper A", "page": 4, "text": "agentic rag improved accuracy by 12 percent over standard rag"}
B = {"title": "Paper B", "page": 7, "text": "agentic rag did not improve accuracy compared with standard rag"}


def _reply(**fields):
    return lambda prompt, **kw: json.dumps(fields)


def test_grounded_contradiction_is_kept(monkeypatch):
    monkeypatch.setattr(disagree, "ask_ollama", _reply(
        contradiction=True, claim_a="improved accuracy by 12 percent", claim_b="did not improve accuracy",
        explanation="Opposite results."))
    result = disagree.compare_pair("accuracy", A, B)
    assert result["a"]["claim"] == "improved accuracy by 12 percent" and result["b"]["page"] == 7


def test_invented_quote_is_rejected(monkeypatch):
    monkeypatch.setattr(disagree, "ask_ollama", _reply(
        contradiction=True, claim_a="improved accuracy by 12 percent", claim_b="latency doubled in production",
        explanation="x"))
    assert disagree.compare_pair("accuracy", A, B) is None


def test_no_contradiction_and_bad_json(monkeypatch):
    monkeypatch.setattr(disagree, "ask_ollama", _reply(contradiction=False))
    assert disagree.compare_pair("accuracy", A, B) is None
    monkeypatch.setattr(disagree, "ask_ollama", lambda prompt, **kw: "they seem to disagree")
    assert disagree.compare_pair("accuracy", A, B) is None


def test_choose_themes_dedupes_and_caps():
    rows = [{"subtopic": "accuracy", "status": "covered"}, {"subtopic": "cost", "status": "thin"}]
    comparison = {"shared": {"metrics": [{"label": "accuracy"}, {"label": "F1"}], "methods": []}}
    assert disagree.choose_themes(rows, comparison) == ["accuracy", "F1"]


def test_find_disagreements_checks_each_pair_once(monkeypatch, tmp_path):
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    col = store.get_collection(str(tmp_path / "chroma"))
    papers = []
    for i, text in enumerate([A["text"], B["text"], "accuracy was similar for both systems"]):
        url = f"u{i}"
        papers.append({"title": f"Paper {i}", "pdf_url": url})
        store.add_chunks(col, [{"chunk_id": f"c{i}", "text": text, "page": 1, "title": f"Paper {i}",
                                "authors": "", "published": "", "url": "", "pdf_url": url}])
    monkeypatch.setattr(disagree, "ask_ollama", _reply(contradiction=False))
    found, checked = disagree.find_disagreements(col, papers, ["accuracy"])
    assert checked == 3 and found == []  # 3 papers -> 3 pairs
