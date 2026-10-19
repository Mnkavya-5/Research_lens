from src.gaps.extract import ASPECTS, _words


def _key(text: str) -> str:
    return " ".join(sorted(_words(text)))  # "RAGAS metrics" and "metrics, RAGAS" -> same key


def compare_papers(extractions: dict[str, dict]) -> dict:
    """extractions: {paper_title: extract_aspects(...) result}. Pure code, no LLM: easy to check."""
    matrix = {aspect: {} for aspect in ASPECTS}  # aspect -> key -> {"label", "papers": {title: page}}
    for title, ex in extractions.items():
        for aspect in ASPECTS:
            for item in ex.get(aspect, []):
                entry = matrix[aspect].setdefault(_key(item["text"]), {"label": item["text"], "papers": {}})
                entry["papers"][title] = item["page"]

    shared = {a: [e for e in items.values() if len(e["papers"]) >= 2] for a, items in matrix.items()}
    missing = {a: [t for t, ex in extractions.items() if not ex.get(a) and not ex.get("error")] for a in ASPECTS}
    return {"matrix": matrix, "shared": shared, "missing": missing, "n_papers": len(extractions)}
