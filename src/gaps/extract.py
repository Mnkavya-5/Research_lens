import json
import re

import requests

from src.llm.ollama_client import ask_ollama
from src.retrieval import embeddings

ASPECTS = {
    "methods": "proposed method, approach, architecture or technique",
    "datasets": "dataset, benchmark or corpus used for experiments",
    "metrics": "evaluation metric or measure, such as accuracy, faithfulness or latency",
    "limitations": "limitations, weaknesses, threats to validity or future work",
}
STOPWORDS = {"a", "an", "the", "of", "for", "and", "or", "in", "on", "to", "with", "we", "our", "is", "are", "by"}

PROMPT = """Passages from the paper "{title}":
{passages}

From these passages only, list the paper's:
- methods: {methods}
- datasets: {datasets}
- metrics: {metrics}
- limitations: {limitations}
Copy short phrases (2-8 words) exactly as written in the passages. Use [] when the passages do not mention one.
Reply with JSON only: {{"methods": [], "datasets": [], "metrics": [], "limitations": []}}"""


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if w not in STOPWORDS}


def is_grounded(item: str, evidence: str, min_overlap: float = 0.8) -> bool:
    """True if most content words of the item appear in the evidence (blocks invented items)."""
    words = _words(item)
    return bool(words) and len(words & _words(evidence)) / len(words) >= min_overlap


def paper_passages(collection, paper: dict, per_aspect: int = 2) -> list[dict]:
    seen, passages = set(), []
    for description in ASPECTS.values():
        result = collection.query(query_embeddings=embeddings.embed([description]), n_results=per_aspect,
                                  where={"pdf_url": paper["pdf_url"]})
        for cid, doc, meta in zip(result["ids"][0], result["documents"][0], result["metadatas"][0]):
            if cid not in seen:
                seen.add(cid)
                passages.append({"text": doc, "page": meta["page"]})
    return passages


def _find_page(item: str, passages: list[dict]) -> int | None:
    best = max(passages, key=lambda p: len(_words(item) & _words(p["text"])), default=None)
    return best["page"] if best else None


def extract_aspects(collection, paper: dict) -> dict:
    """Returns {"methods": [{"text", "page"}], ...} with only items found in the paper's own text."""
    passages = paper_passages(collection, paper)
    result = {aspect: [] for aspect in ASPECTS}
    if not passages:
        return {**result, "error": "paper is not indexed"}

    evidence = "\n".join(p["text"] for p in passages)
    prompt = PROMPT.format(title=paper["title"], passages="\n\n".join(f"(p. {p['page']}) {p['text']}" for p in passages),
                           **ASPECTS)
    try:
        reply = ask_ollama(prompt, temperature=0)
        data = json.loads(re.search(r"\{.*\}", reply, re.DOTALL).group(0))
    except (requests.RequestException, AttributeError, ValueError) as e:
        return {**result, "error": f"extraction failed: {e}"}

    dropped = 0
    for aspect in ASPECTS:
        for item in data.get(aspect) or []:
            if not isinstance(item, str):
                continue
            if is_grounded(item, evidence):
                result[aspect].append({"text": item.strip(), "page": _find_page(item, passages)})
            else:
                dropped += 1  # the model wrote something not supported by the passages
    return {**result, "dropped": dropped}
