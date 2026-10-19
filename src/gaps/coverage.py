import time

import requests

from src.discovery.arxiv import search_arxiv
from src.discovery.dedupe import normalize_title
from src.discovery.discover import to_arxiv_query
from src.discovery.query_gen import _clean_lines
from src.gaps.extract import _words
from src.llm.ollama_client import ask_ollama
from src.retrieval import embeddings

PROBE_K = 10  # nearest passages to check for each subtopic
TOPIC_SIMILARITY = 0.45  # same scale as ranking: on-topic papers scored ~0.5-0.7, off-topic ~0.2-0.3

PROMPT = """Research topic: {topic}
List {n} specific sub-topics or aspects that a thorough study of this topic would be expected to cover.
Output one sub-topic per line, 1-3 words each. No numbering, no explanations."""


SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "es", "s", "ed")


def _stem(word: str) -> str:
    # crude stemming: "costs" -> "cost", "evaluated"/"evaluation" -> "evalu"
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    return word[:5]


def _stems(text: str) -> set[str]:
    return {_stem(w) for w in _words(text)}


def mentions(subtopic: str, passage: str) -> bool:
    """Lexical check: the subtopic's words appear in the passage.
    Up to 2 words: all must appear. Longer: at least two thirds."""
    wanted = _stems(subtopic)
    if not wanted:
        return False
    share = len(wanted & _stems(passage)) / len(wanted)
    return share == 1.0 if len(wanted) <= 2 else share >= 0.67


def propose_subtopics(topic: str, n: int = 6) -> list[str]:
    try:
        reply = ask_ollama(PROMPT.format(topic=topic, n=n), temperature=0)
    except requests.RequestException:
        return []
    return _clean_lines(reply)[:n]


def probe_collection(collection, pdf_urls: list[str], subtopic: str) -> dict:
    """Hybrid check within the selected papers: semantic search finds candidates,
    then a word check confirms the passage really mentions the subtopic."""
    result = collection.query(query_embeddings=embeddings.embed([subtopic]), n_results=PROBE_K,
                              where={"pdf_url": {"$in": pdf_urls}})
    papers, evidence = {}, None
    for doc, dist, meta in zip(result["documents"][0], result["distances"][0], result["metadatas"][0]):
        if mentions(subtopic, doc):
            papers.setdefault(meta["title"], meta["page"])
            evidence = evidence or {"passage": doc, "title": meta["title"], "page": meta["page"],
                                    "distance": round(dist, 3)}
    return {"papers": papers, "evidence": evidence}


def external_query(topic: str, subtopic: str) -> str:
    # subtopic words AND at least one topic word: all:latency AND (all:rag OR all:agentic OR ...)
    topic_terms = " OR ".join(to_arxiv_query(topic).split(" AND "))
    return f"{to_arxiv_query(subtopic)} AND ({topic_terms})"


def check_external(topic: str, subtopic: str, max_results: int = 20) -> list[dict]:
    """Quick arXiv search: is there on-topic work on this subtopic that we did not collect?"""
    try:
        papers = search_arxiv(external_query(topic, subtopic), max_results)
    except requests.RequestException:
        return []
    papers = [p for p in papers if mentions(subtopic, f"{p['title']} {p['abstract'] or ''}")]
    if not papers:
        return []
    # keep only papers our ranking would call relevant to the *whole* topic, not just the subtopic
    topic_vec, *vecs = embeddings.embed([topic] + [f"{p['title']}. {p['abstract'] or ''}" for p in papers])
    relevant = []
    for paper, vec in zip(papers, vecs):
        sim = sum(a * b for a, b in zip(topic_vec, vec))
        if sim >= TOPIC_SIMILARITY:
            relevant.append({**paper, "similarity": round(sim, 3)})
    return sorted(relevant, key=lambda p: p["similarity"], reverse=True)


def coverage_report(collection, topic: str, papers: list[dict], subtopics: list[str] | None = None) -> list[dict]:
    pdf_urls = [p["pdf_url"] for p in papers if p.get("pdf_url")]
    rows = []
    searched = 0
    for subtopic in (subtopics if subtopics is not None else propose_subtopics(topic)):
        probe = probe_collection(collection, pdf_urls, subtopic)
        n = len(probe["papers"])
        if n >= 2:
            rows.append({"subtopic": subtopic, "status": "covered", **probe, "external": []})
            continue
        if searched:
            time.sleep(3)  # arXiv rate limit
        searched += 1
        external = check_external(topic, subtopic)
        if n == 1:
            status = "thin"  # only one collected paper mentions it
        else:
            status = "collection gap" if external else "possibly underexplored"
        rows.append({"subtopic": subtopic, "status": status, **probe, "external": external[:3]})
    return rows


def papers_to_add(rows: list[dict], existing: list[dict] = ()) -> list[dict]:
    """On-topic arXiv papers found for collection gaps and thin areas, ready to index."""
    seen = {normalize_title(p["title"]) for p in existing}  # never suggest papers we already have
    out = []
    for r in rows:
        for p in r["external"]:
            key = normalize_title(p["title"])
            if r["status"] in ("collection gap", "thin") and key not in seen:
                seen.add(key)
                out.append(p)
    return out
