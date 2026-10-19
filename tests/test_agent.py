import requests

from src.rag import agent
from src.rag.answer import NOT_FOUND
from src.retrieval import embeddings
from tests.test_rag import fake_embed


def _grounded(answer=NOT_FOUND, sources=()):
    return {"answer": answer, "sources": list(sources), "cited": [], "invalid_citations": [],
            "unsupported": [], "warning": None}


def _src(title, distance, page=1):
    return {"title": title, "distance": distance, "page": page, "url": "u", "pdf_url": "p", "text": "x"}


def test_papers_in_collection_dedupes_and_sorts():
    papers = agent.papers_in_collection([_src("A", 0.5, 2), _src("B", 0.3), _src("A", 0.4, 7)])
    assert [(p["title"], p["page"]) for p in papers] == [("B", 1), ("A", 7)]


def test_general_answer_flags_reference_like_text(monkeypatch):
    monkeypatch.setattr(agent, "ask_ollama", lambda prompt, **kw: "As shown by Smith et al. (2023), RAG helps.")
    assert "not checked" in agent.general_answer("q", NOT_FOUND)["warning"]
    monkeypatch.setattr(agent, "ask_ollama", lambda prompt, **kw: "RAG retrieves documents before answering.")
    assert agent.general_answer("q", NOT_FOUND)["warning"] is None


def test_general_prompt_gets_grounded_answer_without_citation_numbers(monkeypatch):
    prompts = []
    monkeypatch.setattr(agent, "ask_ollama", lambda prompt, **kw: prompts.append(prompt) or "ok")
    agent.general_answer("q", "They used FIQA [2].")
    assert "They used FIQA ." in prompts[0] and "[2]" not in prompts[0]


def test_find_new_papers_excludes_collected_and_off_topic(monkeypatch):
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    monkeypatch.setattr(agent, "generate_queries", lambda q, n: [q, "rag evaluation metrics"])
    monkeypatch.setattr(agent, "search_arxiv", lambda q, n: [
        {"title": "RAG evaluation metrics", "abstract": "rag evaluation metrics benchmark"},
        {"title": "Already Have It", "abstract": "rag evaluation metrics"},
        {"title": "Protein folding", "abstract": "molecular dynamics of proteins"},
    ])
    found = agent.find_new_papers("rag evaluation metrics", {"already have it."})
    assert [p["title"] for p in found] == ["RAG evaluation metrics"]


def test_research_answer_layers_and_survives_failures(monkeypatch):
    monkeypatch.setattr(agent, "answer_question", lambda col, q: _grounded(sources=[_src("A", 0.4)]))
    monkeypatch.setattr(agent, "ask_ollama", lambda prompt, **kw: "General explanation.")
    def arxiv_down(*a, **k):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(agent, "find_new_papers", arxiv_down)
    monkeypatch.setattr(agent, "question_keywords", lambda q: "kw")
    monkeypatch.setattr(agent, "search_web", lambda q, max_results: (
        [{"title": "Blog", "url": "https://blog.example/x", "found_via": "hackernews"}], []))

    result = agent.research_answer(None, "q")
    assert result["web"][0]["title"] == "Blog"
    assert result["general"]["text"] == "General explanation."
    assert result["papers"][0]["title"] == "A"
    assert result["new_papers"] == [] and "arXiv search failed" in result["errors"][0]


def test_strict_mode_skips_general_knowledge(monkeypatch):
    monkeypatch.setattr(agent, "answer_question", lambda col, q: _grounded())
    def must_not_call(*a, **k):
        raise AssertionError("general knowledge should be off")
    monkeypatch.setattr(agent, "ask_ollama", must_not_call)
    result = agent.research_answer(None, "q", use_general_knowledge=False, search_new_papers=False,
                                   search_web_pages=False)
    assert result["general"] is None


def test_find_new_papers_searches_llm_keywords_not_raw_question(monkeypatch):
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    monkeypatch.setattr(agent, "generate_queries", lambda q, n: [q, "agentic RAG evaluation"])
    queries = []
    monkeypatch.setattr(agent, "search_arxiv", lambda q, n: queries.append(q) or [])
    agent.find_new_papers("What are the main ways to evaluate agentic RAG systems?", set())
    assert queries == ["all:agentic AND all:rag AND all:evaluation"]


def test_web_suggestions_skip_pages_already_indexed(monkeypatch, tmp_path):
    from src.retrieval import store
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    col = store.get_collection(str(tmp_path / "chroma"))
    store.add_chunks(col, [{"chunk_id": "w1", "text": "rag reduces hallucinations", "page": 1, "title": "Old",
                            "authors": "", "published": "", "url": "https://old.example", "pdf_url": "", "kind": "web"}])
    monkeypatch.setattr(agent, "ask_ollama", lambda prompt, **kw: "ok")
    monkeypatch.setattr(agent, "question_keywords", lambda q: "kw")
    monkeypatch.setattr(agent, "search_web", lambda q, max_results: ([
        {"title": "Old", "url": "https://old.example", "found_via": "hackernews"},
        {"title": "New", "url": "https://new.example", "found_via": "hackernews"}], []))
    result = agent.research_answer(col, "rag?", search_new_papers=False)
    assert [w["title"] for w in result["web"]] == ["New"]
