import pymupdf
import pytest

from src.processing.pdf import ExtractionError, extract_pages

LONG_TEXT = "Retrieval augmented generation is evaluated on faithfulness. " * 5


def _make_pdf(path, page_texts):
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(72, 72, 540, 770), text)
    doc.save(path)
    doc.close()
    return path


def test_pages_numbered_from_one(tmp_path):
    pdf = _make_pdf(tmp_path / "a.pdf", [LONG_TEXT, "Page two " + LONG_TEXT])
    pages = extract_pages(pdf)
    assert [p["page"] for p in pages] == [1, 2]
    assert "Page two" in pages[1]["text"]


def test_blank_pdf_flagged_as_scanned(tmp_path):
    pdf = _make_pdf(tmp_path / "scan.pdf", ["", ""])
    with pytest.raises(ExtractionError, match="scanned"):
        extract_pages(pdf)


def test_corrupt_file_raises_extraction_error(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.7 this is not really a pdf")
    with pytest.raises(ExtractionError):
        extract_pages(bad)
