from pathlib import Path

import pymupdf

MIN_CHARS_PER_PAGE = 100  # below this on average, the PDF is probably scanned images


class ExtractionError(Exception):
    """A PDF that we could not turn into usable text."""


def extract_pages(pdf_path: str | Path) -> list[dict]:
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:  # PyMuPDF raises several error types for broken files
        raise ExtractionError(f"cannot open {pdf_path}: {e}") from e

    with doc:
        if doc.needs_pass:
            raise ExtractionError(f"{pdf_path} is password-protected")
        pages = [{"page": i + 1, "text": page.get_text("text")}
                 for i, page in enumerate(doc)]

    total_chars = sum(len(p["text"].strip()) for p in pages)
    if not pages or total_chars / len(pages) < MIN_CHARS_PER_PAGE:
        raise ExtractionError(f"{pdf_path} has almost no text (probably scanned; needs OCR)")
    return pages
