import json
import re
from itertools import combinations

import requests

from src.gaps.coverage import mentions
from src.gaps.extract import is_grounded
from src.llm.ollama_client import ask_ollama
from src.retrieval import embeddings

MAX_THEMES = 5
MAX_PAPERS_PER_THEME = 3  # 3 papers -> 3 pairs; keeps LLM calls (and time) bounded

PROMPT = """Theme: {theme}

Passage A ({title_a}, p. {page_a}):
{text_a}

Passage B ({title_b}, p. {page_b}):
{text_b}

Do passages A and B make claims about the theme that contradict each other
(for example, one reports an improvement and the other reports no improvement or a decline)?
Different topics, different details, or results on different tasks are NOT contradictions.
Reply with JSON only:
{{"contradiction": true or false, "claim_a": "<exact quote from A>", "claim_b": "<exact quote from B>", "explanation": "<one sentence>"}}"""


def passage_for(collection, paper: dict, theme: str) -> dict | None:
    """The passage in this paper closest to the theme that actually mentions it."""
    result = collection.query(query_embeddings=embeddings.embed([theme]), n_results=3,
                              where={"pdf_url": paper["pdf_url"]})
    for doc, meta in zip(result["documents"][0], result["metadatas"][0]):
        if mentions(theme, doc):
            return {"title": paper["title"], "page": meta["page"], "text": doc}
    return None


def compare_pair(theme: str, a: dict, b: dict) -> dict | None:
    prompt = PROMPT.format(theme=theme, title_a=a["title"], page_a=a["page"], text_a=a["text"],
                           title_b=b["title"], page_b=b["page"], text_b=b["text"])
    try:
        reply = ask_ollama(prompt, temperature=0)
        data = json.loads(re.search(r"\{.*\}", reply, re.DOTALL).group(0))
    except (requests.RequestException, AttributeError, ValueError):
        return None
    if data.get("contradiction") is not True:
        return None
    claim_a, claim_b = str(data.get("claim_a", "")), str(data.get("claim_b", ""))
    # both quotes must really be in their passages, or we do not report it
    if not (is_grounded(claim_a, a["text"]) and is_grounded(claim_b, b["text"])):
        return None
    return {"theme": theme, "a": {**a, "claim": claim_a}, "b": {**b, "claim": claim_b},
            "explanation": str(data.get("explanation", ""))}


def choose_themes(coverage_rows: list[dict], comparison: dict) -> list[str]:
    themes = [r["subtopic"] for r in coverage_rows if r["status"] == "covered"]
    for aspect in ("metrics", "methods"):
        themes += [e["label"] for e in comparison["shared"].get(aspect, [])]
    return list(dict.fromkeys(themes))[:MAX_THEMES]  # dict.fromkeys removes duplicates, keeps order


def find_disagreements(collection, papers: list[dict], themes: list[str]) -> tuple[list[dict], int]:
    """Returns (possible disagreements, number of pairs checked)."""
    found, checked = [], 0
    for theme in themes:
        passages = [p for p in (passage_for(collection, paper, theme) for paper in papers) if p]
        for a, b in combinations(passages[:MAX_PAPERS_PER_THEME], 2):
            checked += 1
            result = compare_pair(theme, a, b)
            if result:
                found.append(result)
    return found, checked
