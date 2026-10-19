import json
import os
import re

import requests

from src.llm.ollama_client import ask_ollama
from src.retrieval import embeddings

JUDGE_MODEL = os.getenv("JUDGE_MODEL")  # optional: a bigger model just for judging, e.g. qwen3:4b

PROMPT = """Research topic: {topic}

Paper: {title}
Evidence ({evidence_type}):
{evidence}

Rate how useful this paper is for researching the topic, based only on the evidence above.
5 = directly studies the topic with methods and results; 3 = partly related; 1 = unrelated.
Reply with JSON only: {{"score": <1-5>, "reason": "<one short sentence>"}}"""


def gather_evidence(topic: str, paper: dict, collection=None, k: int = 3) -> tuple[str, str]:
    """Prefer full-text passages from the indexed PDF; fall back to the abstract."""
    if collection is not None and paper.get("pdf_url"):
        result = collection.query(query_embeddings=embeddings.embed([topic]), n_results=k,
                                  where={"pdf_url": paper["pdf_url"]})
        docs = result["documents"][0]
        if docs:
            pages = [m["page"] for m in result["metadatas"][0]]
            text = "\n\n".join(f"(p. {p}) {d}" for p, d in zip(pages, docs))
            return text, f"full text, {len(docs)} passages"
    if paper.get("abstract"):
        return paper["abstract"], "abstract only"
    return paper["title"], "title only"


def parse_judgement(reply: str) -> tuple[int | None, str]:
    match = re.search(r"\{.*\}", reply, re.DOTALL)  # tolerate text around the JSON
    if not match:
        return None, "could not read the judge's reply"
    try:
        data = json.loads(match.group(0))
        score = int(data["score"])
    except (ValueError, KeyError, TypeError):
        return None, "could not read the judge's reply"
    if not 1 <= score <= 5:
        return None, f"judge gave an out-of-range score ({score})"
    return score, str(data.get("reason", "")).strip()


def judge_papers(topic: str, papers: list[dict], collection=None) -> list[dict]:
    judged = []
    for paper in papers:
        evidence, evidence_type = gather_evidence(topic, paper, collection)
        prompt = PROMPT.format(topic=topic, title=paper["title"],
                               evidence_type=evidence_type, evidence=evidence)
        try:
            score, reason = parse_judgement(ask_ollama(prompt, temperature=0, model=JUDGE_MODEL))
        except requests.RequestException as e:
            score, reason = None, f"judge unavailable: {e}"
        judged.append({**paper, "llm_score": score, "llm_reason": reason, "evidence_type": evidence_type})
    return judged


def combined_score(paper: dict) -> float:
    """Average of baseline (0-1) and judge (1-5 scaled to 0-1). Baseline alone if the judge failed."""
    if paper.get("llm_score") is None:
        return paper["score"]
    return round(0.5 * paper["score"] + 0.5 * (paper["llm_score"] - 1) / 4, 3)
