"""Evaluate ResearchLens: retrieval, answers, refusals and ranking.

Run from the project root (Ollama must be running):
    python -m eval.run_eval
Writes eval/results.md and prints a summary.
"""
import json
import os
import time
from datetime import date
from pathlib import Path

from src.evaluation import metrics as m
from src.rag.answer import NOT_FOUND, answer_question
from src.ranking.baseline import WEIGHTS, rank_papers
from src.retrieval.store import get_collection, search

EVAL_DIR = Path(__file__).parent
K = int(os.getenv("EVAL_K", "8"))  # chunks retrieved per question


def evaluate_qa(collection) -> list[dict]:
    rows = []
    for q in json.loads((EVAL_DIR / "qa_set.json").read_text())["questions"]:
        start = time.time()
        retrieved = search(collection, q["question"], K)
        result = answer_question(collection, q["question"], K)
        answer = result["answer"]
        row = {"question": q["question"], "answerable": q["answerable"], "answer": answer,
               "refused": NOT_FOUND in answer, "warned": bool(result["warning"]),
               "seconds": round(time.time() - start, 1)}

        if q["answerable"]:
            gold = lambda c: q["paper"].lower() in c["title"].lower()
            rank = m.first_hit_rank(retrieved, gold)
            support = m.sentence_support(answer, result["sources"])
            row.update({
                "paper_rank": rank,
                "evidence_retrieved": any(q["evidence"].lower() in c["text"].lower() for c in retrieved),
                "correct": m.keyword_match(answer, q["answer_keywords"], q["min_match"]),
                "cites_gold_paper": any(gold(result["sources"][n - 1]) for n in result["cited"]),
                "invalid_citations": result["invalid_citations"],
                "sentences": len(support),
                "supported_sentences": sum(r["supported"] for r in support),
            })
        rows.append(row)
        print(f"  {row['seconds']:5.1f}s | {'refused' if row['refused'] else 'answered'} | {q['question'][:70]}")
    return rows


def evaluate_ranking() -> dict:
    data = json.loads((EVAL_DIR / "ranking_set.json").read_text())
    topic, papers = data["topic"], data["papers"]
    variants = {
        "similarity only": dict(weights={"relevance": 1.0, "recency": 0, "open_pdf": 0, "has_abstract": 0}, normalize=False),
        "weights, raw similarity (current default)": dict(weights=WEIGHTS, normalize=False),
        "weights, min-max relevance (tried, rejected)": dict(weights=WEIGHTS, normalize=True),
    }
    out = {}
    for name, opts in variants.items():
        grades = [p["relevance"] for p in rank_papers(topic, papers, **opts)]
        out[name] = {"P@5": m.precision_at_k(grades, 5), "P@10": m.precision_at_k(grades, 10),
                     "nDCG@10": round(m.ndcg_at_k(grades, 10), 3)}
    return out


def summarise(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    answered = [r for r in ans if not r["refused"]]
    pct = lambda n, d: round(100 * n / d) if d else 0
    return {
        f"Retrieval hit@{K} (gold paper in top {K})": pct(sum(r["paper_rank"] is not None for r in ans), len(ans)),
        "Retrieval MRR": round(sum(m.reciprocal_rank(r["paper_rank"]) for r in ans) / len(ans), 2),
        f"Evidence recall@{K} (answer text retrieved)": pct(sum(r["evidence_retrieved"] for r in ans), len(ans)),
        "Answer correctness": pct(sum(r["correct"] for r in ans), len(ans)),
        "Cites the gold paper (answered questions)": pct(sum(r["cites_gold_paper"] for r in answered), len(answered)),
        "Faithfulness (supported sentences)": pct(sum(r["supported_sentences"] for r in answered),
                                                  sum(r["sentences"] for r in answered)),
        "Invalid citations": sum(len(r["invalid_citations"]) for r in ans),
        "False refusals (answerable but refused)": pct(sum(r["refused"] for r in ans), len(ans)),
        "Correct refusals (unanswerable)": pct(sum(r["refused"] for r in unans), len(unans)),
        "Unanswerable: refused OR warned": pct(sum(r["refused"] or r["warned"] for r in unans), len(unans)),
        "Answered questions with a warning": pct(sum(r["warned"] for r in answered), len(answered)),
    }


def write_report(summary: dict, ranking: dict, rows: list[dict]) -> Path:
    lines = [f"# ResearchLens evaluation ({date.today()}, k={K})", "",
             f"{sum(r['answerable'] for r in rows)} answerable + {sum(not r['answerable'] for r in rows)} unanswerable "
             "questions; labels written by an AI assistant from the papers (see qa_set.json).", "",
             "## Q&A metrics (percent unless noted)", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary.items()]
    lines += ["", "## Ranking (48 labelled papers)", "", "| Variant | P@5 | P@10 | nDCG@10 |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['P@5']:.2f} | {v['P@10']:.2f} | {v['nDCG@10']:.2f} |" for k, v in ranking.items()]
    lines += ["", "## Per question", "", "| Question | Rank | Evidence | Correct | Refused | Supported | Answer |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        ans = r["answer"].replace("|", "/").replace("\n", " ")[:160]
        if r["answerable"]:
            lines.append(f"| {r['question'][:60]} | {r['paper_rank']} | {r['evidence_retrieved']} | {r['correct']} | "
                         f"{r['refused']} | {r['supported_sentences']}/{r['sentences']} | {ans} |")
        else:
            lines.append(f"| {r['question'][:60]} (unanswerable) | - | - | - | {r['refused']} | - | {ans} |")
    path = EVAL_DIR / f"results_k{K}.md"
    path.write_text("\n".join(lines) + "\n")
    return path


if __name__ == "__main__":
    print("Q&A evaluation:")
    rows = evaluate_qa(get_collection())
    summary = summarise(rows)
    ranking = evaluate_ranking()
    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nRanking:")
    for k, v in ranking.items():
        print(f"  {k}: {v}")
    print(f"\nReport: {write_report(summary, ranking, rows)}")
