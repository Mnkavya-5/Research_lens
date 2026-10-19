import re
from datetime import date

from src.retrieval import embeddings

# Transparent weights: every score can be explained as a sum of these parts.
WEIGHTS = {"relevance": 0.60, "recency": 0.20, "open_pdf": 0.15, "has_abstract": 0.05}
RECENCY_YEARS = 10  # a paper this old or older gets 0 for recency


def paper_year(paper: dict) -> int | None:
    match = re.match(r"(\d{4})", paper.get("published") or "")
    return int(match.group(1)) if match else None


def recency_score(year: int | None, this_year: int | None = None) -> float:
    if year is None:
        return 0.0
    age = (this_year or date.today().year) - year
    return max(0.0, 1 - age / RECENCY_YEARS)


def _paper_text(paper: dict) -> str:
    abstract = " ".join((paper.get("abstract") or "").split()[:120])
    return f"{paper['title']}. {abstract}"


def rank_papers(topic: str, papers: list[dict], weights: dict = WEIGHTS, normalize: bool = False) -> list[dict]:
    if not papers:
        return []
    topic_vec, *paper_vecs = embeddings.embed([topic] + [_paper_text(p) for p in papers])
    sims = [sum(a * b for a, b in zip(topic_vec, v)) for v in paper_vecs]  # cosine, vectors are normalized

    # normalize=True (min-max scaling within the result set) looked better by eye but scored worse in
    # evaluation (nDCG@10 0.50 vs 0.67): it amplified title-only papers, whose short text gets
    # inflated similarity. Kept as an option for experiments; off by default.
    lo, hi = min(sims), max(sims)
    ranked = []
    for paper, sim in zip(papers, sims):
        parts = {
            "relevance": ((sim - lo) / (hi - lo) if hi > lo else 1.0) if normalize else max(0.0, sim),
            "recency": recency_score(paper_year(paper)),
            "open_pdf": 1.0 if paper.get("pdf_url") else 0.0,
            "has_abstract": 1.0 if paper.get("abstract") else 0.0,
        }
        score = sum(weights[k] * v for k, v in parts.items())
        ranked.append({**paper, "score": round(score, 3), "score_parts": parts, "similarity": round(sim, 3)})
    return sorted(ranked, key=lambda p: p["score"], reverse=True)
