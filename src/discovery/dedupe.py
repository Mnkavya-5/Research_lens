import re


def normalize_title(title: str) -> str:
    text = title.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def _richness(paper: dict) -> int:
    # A PDF is worth more than an abstract: we can only chunk full text.
    return 2 * bool(paper.get("pdf_url")) + bool(paper.get("abstract"))


def dedupe_papers(papers: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for paper in papers:
        key = normalize_title(paper["title"])
        if key not in best:
            best[key] = paper
            continue
        keep, other = sorted([best[key], paper], key=_richness, reverse=True)
        merged = dict(keep)
        for field, value in other.items():
            if not merged.get(field) and value:
                merged[field] = value  # fill gaps from the poorer copy
        best[key] = merged
    return list(best.values())
