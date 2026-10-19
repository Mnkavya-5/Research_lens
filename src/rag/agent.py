import re

import requests

from src.discovery.arxiv import search_arxiv
from src.discovery.dedupe import normalize_title
from src.discovery.discover import to_arxiv_query
from src.discovery.query_gen import generate_queries
from src.llm.ollama_client import ask_ollama
from src.rag.answer import NOT_FOUND, answer_question
from src.retrieval import embeddings
from src.web.search import search_web

NEW_PAPER_SIMILARITY = 0.28  # question-vs-abstract: relevant papers scored 0.29-0.68, junk 0.01-0.26 (small sample)

GENERAL_PROMPT = """Question: {question}

What the user's collected papers say (may be incomplete or empty):
{grounded}

Now answer the question as an expert research assistant, using your own general knowledge:
explain the key ideas, compare approaches, point out trade-offs, and suggest what to explore next.
Do not repeat the summary above. Do not invent paper titles, authors, numbers or citations.
If you are unsure about something, say so. Write 4-8 sentences."""

# "(Smith et al., 2023)", "Smith et al. (2023)", "[3]": references the model may have made up
REFERENCE_LIKE = re.compile(r"et al\.?|\(\s*[A-Z][a-z]+,?\s+\d{4}\s*\)|\[\d+\]")


def papers_in_collection(sources: list[dict]) -> list[dict]:
    """The collected papers behind the retrieved passages, best match first."""
    best = {}
    for s in sources:
        if s["title"] not in best or s["distance"] < best[s["title"]]["distance"]:
            best[s["title"]] = {"title": s["title"], "url": s.get("url", ""), "pdf_url": s.get("pdf_url"),
                                "page": s["page"], "distance": s["distance"]}
    return sorted(best.values(), key=lambda p: p["distance"])


def question_keywords(question: str) -> str:
    # Questions contain words like "main" or "ways" that break keyword searches, so let the LLM
    # turn the question into short keywords first (e.g. "agentic RAG evaluation").
    keywords = generate_queries(question, n=1)[1:]  # [0] is the question itself
    return keywords[0] if keywords else question


def find_new_papers(question: str, existing_titles: set[str], max_results: int = 15, keep: int = 3,
                    keywords: str | None = None) -> list[dict]:
    """Quick arXiv search for the question; keeps on-topic papers we do not have yet."""
    query = to_arxiv_query(keywords or question_keywords(question))
    if not query:
        return []
    papers = search_arxiv(query, max_results)
    seen = {normalize_title(t) for t in existing_titles}
    papers = [p for p in papers if normalize_title(p["title"]) not in seen]
    if not papers:
        return []
    q_vec, *vecs = embeddings.embed([question] + [f"{p['title']}. {p['abstract'] or ''}" for p in papers])
    scored = [{**p, "similarity": round(sum(a * b for a, b in zip(q_vec, v)), 3)} for p, v in zip(papers, vecs)]
    return sorted([p for p in scored if p["similarity"] >= NEW_PAPER_SIMILARITY],
                  key=lambda p: p["similarity"], reverse=True)[:keep]


def already_indexed(collection, url: str) -> bool:
    if collection is None:
        return False
    return bool(collection.get(where={"url": url}, limit=1)["ids"])


def general_answer(question: str, grounded_answer: str) -> dict:
    grounded = "" if NOT_FOUND in grounded_answer else re.sub(r"\[\d+\]", "", grounded_answer)
    text = ask_ollama(GENERAL_PROMPT.format(question=question, grounded=grounded or "(nothing relevant found)"),
                      temperature=0).strip()
    warning = None
    if REFERENCE_LIKE.search(text):
        warning = "mentions references or citations that were not checked; they may not exist"
    return {"text": text, "warning": warning}


def research_answer(collection, question: str, use_general_knowledge: bool = True,
                    search_new_papers: bool = True, search_web_pages: bool = True) -> dict:
    grounded = answer_question(collection, question)  # unchanged, evaluated pipeline
    result = {"grounded": grounded, "general": None, "papers": papers_in_collection(grounded["sources"]),
              "new_papers": [], "web": [], "errors": []}

    if use_general_knowledge:
        try:
            result["general"] = general_answer(question, grounded["answer"])
        except requests.RequestException as e:
            result["errors"].append(f"general-knowledge answer failed: {e}")

    if search_new_papers or search_web_pages:
        keywords = question_keywords(question)
    if search_new_papers:
        try:
            result["new_papers"] = find_new_papers(question, {p["title"] for p in result["papers"]},
                                                   keywords=keywords)
        except requests.RequestException as e:
            result["errors"].append(f"arXiv search failed: {e}")
    if search_web_pages:
        pages, errors = search_web(keywords, max_results=5)
        result["web"] = [p for p in pages if not already_indexed(collection, p["url"])][:3]
        result["errors"] += errors
    return result
