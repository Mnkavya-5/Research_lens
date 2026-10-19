import pytest

from src.gaps import compare, coverage, extract
from src.retrieval import embeddings, store
from tests.test_rag import fake_embed

PAPER = {"title": "RAG Eval", "pdf_url": "https://arxiv.org/pdf/1"}


@pytest.fixture
def collection(monkeypatch, tmp_path):
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    col = store.get_collection(str(tmp_path / "chroma"))
    store.add_chunks(col, [
        {"chunk_id": "1_p3_c0", "text": "we evaluate on the HotpotQA dataset using faithfulness and latency metrics",
         "page": 3, "title": "RAG Eval", "authors": "", "published": "", "url": "", "pdf_url": PAPER["pdf_url"]},
        {"chunk_id": "1_p9_c0", "text": "a limitation is the small sample of 85 questions",
         "page": 9, "title": "RAG Eval", "authors": "", "published": "", "url": "", "pdf_url": PAPER["pdf_url"]},
    ])
    return col


def test_grounding_check():
    evidence = "We evaluate on the HotpotQA dataset using faithfulness metrics."
    assert extract.is_grounded("HotpotQA dataset", evidence)
    assert not extract.is_grounded("MS MARCO benchmark", evidence)  # invented
    assert not extract.is_grounded("", evidence)


def test_extraction_drops_invented_items_and_keeps_pages(monkeypatch, collection):
    reply = '{"methods": [], "datasets": ["HotpotQA dataset", "Natural Questions"], ' \
            '"metrics": ["faithfulness"], "limitations": ["small sample of 85 questions"]}'
    monkeypatch.setattr(extract, "ask_ollama", lambda prompt, **kw: reply)
    result = extract.extract_aspects(collection, PAPER)
    assert [d["text"] for d in result["datasets"]] == ["HotpotQA dataset"]
    assert result["dropped"] == 1  # "Natural Questions" is not in the paper's text
    assert result["limitations"][0]["page"] == 9


def test_extraction_survives_bad_llm_reply(monkeypatch, collection):
    monkeypatch.setattr(extract, "ask_ollama", lambda prompt, **kw: "I cannot do that.")
    assert "error" in extract.extract_aspects(collection, PAPER)


def test_compare_finds_shared_and_missing():
    ex = {
        "A": {"datasets": [{"text": "HotpotQA", "page": 2}], "metrics": [], "methods": [], "limitations": []},
        "B": {"datasets": [{"text": "hotpotqa", "page": 5}], "metrics": [{"text": "F1", "page": 4}],
              "methods": [], "limitations": []},
    }
    result = compare.compare_papers(ex)
    shared = result["shared"]["datasets"][0]
    assert shared["papers"] == {"A": 2, "B": 5}
    assert result["missing"]["metrics"] == ["A"]
    assert result["missing"]["limitations"] == ["A", "B"]


def test_mentions_requires_every_word():
    assert coverage.mentions("human evaluation", "we ran a human study and evaluated answers")
    assert not coverage.mentions("multilingual evaluation", "we evaluate answers in English")
    assert coverage.mentions("cost", "the costs were high")  # stemming
    assert coverage.mentions("evaluation metrics for agents", "agent evaluation used three metrics")  # 2/3 rule


def test_coverage_labels(monkeypatch, tmp_path):
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    monkeypatch.setattr(coverage.time, "sleep", lambda s: None)
    col = store.get_collection(str(tmp_path / "chroma"))
    def chunk(cid, text, url, title):
        return {"chunk_id": cid, "text": text, "page": 1, "title": title, "authors": "", "published": "",
                "url": "", "pdf_url": url}
    store.add_chunks(col, [
        chunk("a", "we report faithfulness metrics", "u1", "Paper A"),
        chunk("b", "faithfulness metrics and latency", "u2", "Paper B"),
        chunk("c", "latency was measured", "u3", "Unselected paper"),
    ])
    monkeypatch.setattr(coverage, "check_external",
                        lambda topic, sub: [{"title": "Cost of RAG"}] if sub in ("cost", "latency") else [])
    papers = [{"pdf_url": "u1"}, {"pdf_url": "u2"}]
    rows = coverage.coverage_report(col, "rag", papers,
                                    subtopics=["faithfulness metrics", "latency", "cost", "quantum"])
    status = {r["subtopic"]: r["status"] for r in rows}
    assert status == {"faithfulness metrics": "covered", "latency": "thin",
                      "cost": "collection gap", "quantum": "possibly underexplored"}
    assert [p["title"] for p in coverage.papers_to_add(rows)] == ["Cost of RAG"]  # deduplicated
    assert coverage.papers_to_add(rows, existing=[{"title": "cost of RAG."}]) == []  # already collected


def test_external_query_requires_a_topic_word():
    assert coverage.external_query("agentic RAG", "latency") == "all:latency AND (all:agentic OR all:rag)"


def test_external_check_drops_off_topic_papers(monkeypatch):
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    found = [{"title": "medical image segmentation", "abstract": "segmentation of scans"},
             {"title": "agentic rag segmentation", "abstract": "agentic rag pipelines with segmentation"}]
    monkeypatch.setattr(coverage, "search_arxiv", lambda q, n: found)
    kept = coverage.check_external("agentic rag pipelines", "segmentation")
    assert [p["title"] for p in kept] == ["agentic rag segmentation"]
