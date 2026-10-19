import math
import re

from src.gaps.extract import _words


# ---------- retrieval ----------
def first_hit_rank(results: list[dict], is_relevant) -> int | None:
    """1-based rank of the first relevant result, or None if none is relevant."""
    for rank, item in enumerate(results, start=1):
        if is_relevant(item):
            return rank
    return None


def reciprocal_rank(rank: int | None) -> float:
    return 1 / rank if rank else 0.0  # rank 1 -> 1.0, rank 2 -> 0.5, not found -> 0


# ---------- ranking ----------
def precision_at_k(grades: list[int], k: int, min_grade: int = 2) -> float:
    top = grades[:k]
    return sum(g >= min_grade for g in top) / len(top) if top else 0.0


def ndcg_at_k(grades: list[int], k: int) -> float:
    """Graded ranking quality: 1.0 = best possible order, 0 = nothing relevant in the top k."""
    def dcg(gs):
        return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gs[:k]))
    ideal = dcg(sorted(grades, reverse=True))
    return dcg(grades) / ideal if ideal else 0.0


# ---------- answers ----------
def _norm(text: str) -> str:
    return text.lower().replace(",", "")  # "5,000" and "5000" match


def keyword_match(answer: str, keywords: list[str], min_match: int) -> bool:
    return sum(_norm(k) in _norm(answer) for k in keywords) >= min_match


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(_words(s)) >= 3]


def sentence_support(answer: str, sources: list[dict], min_overlap: float = 0.5) -> list[dict]:
    """Faithfulness proxy: is each sentence's wording found in the source(s) it cites?
    A sentence with no citation counts as unsupported."""
    rows = []
    for sentence in split_sentences(answer):
        cited = [int(n) for n in re.findall(r"\[(\d+)\]", sentence) if 1 <= int(n) <= len(sources)]
        words = _words(re.sub(r"\[\d+\]", "", sentence))
        overlap = max((len(words & _words(sources[n - 1]["text"])) / len(words) for n in cited), default=0.0)
        rows.append({"sentence": sentence, "cited": cited, "overlap": round(overlap, 2),
                     "supported": bool(cited) and overlap >= min_overlap})
    return rows
