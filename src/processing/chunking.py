import re

from src.processing.download import pdf_filename

CHUNK_WORDS = 150    # all-MiniLM-L6-v2 reads ~256 tokens (~180 words); longer text is cut off
OVERLAP_WORDS = 30   # so a sentence on a chunk boundary appears whole in at least one chunk
MIN_WORDS = 20       # skip tiny leftovers like page numbers or footers
REFERENCES = re.compile(r"\n\s*(references|bibliography)\s*\n", re.IGNORECASE)


def clean_text(text: str) -> str:
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)  # "of-\nten" -> "often"
    text = re.sub(r"(\w)-\n(\w)", r"\1-\2", text)     # "Domain-\nSpecific" keeps its hyphen
    text = re.sub(r"\s*\n\s*\n\s*", "\n\n", text)   # keep paragraph breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)    # single line breaks -> spaces
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_words(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list[str]:
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if len(piece) >= MIN_WORDS:
            chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def chunk_paper(paper: dict, pages: list[dict]) -> list[dict]:
    paper_id = pdf_filename(paper["pdf_url"]).removesuffix(".pdf")
    chunks = []
    for page in pages:
        raw = page["text"]
        refs = REFERENCES.search(raw)
        if refs:
            raw = raw[:refs.start()]  # citations list adds noise to retrieval
        for n, text in enumerate(split_words(clean_text(raw))):
            chunks.append({
                "chunk_id": f"{paper_id}_p{page['page']}_c{n}",
                "text": text,
                "page": page["page"],
                "title": paper["title"],
                "authors": ", ".join(paper.get("authors") or []),
                "published": paper.get("published") or "",
                "url": paper.get("url") or "",
                "pdf_url": paper["pdf_url"],
            })
        if refs:
            break  # everything after the references heading is skipped
    return chunks
