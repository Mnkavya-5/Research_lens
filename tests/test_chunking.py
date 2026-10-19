from src.processing.chunking import chunk_paper, clean_text, split_words

PAPER = {"title": "RAG Survey", "authors": ["A. One", "B. Two"],
         "pdf_url": "https://arxiv.org/pdf/2511.22858v1", "url": "https://arxiv.org/abs/2511.22858v1"}


def _words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_clean_rejoins_hyphenated_words():
    assert clean_text("systems of-\nten fail") == "systems often fail"


def test_clean_keeps_hyphen_before_capital():
    assert clean_text("Domain-\nSpecific terms") == "Domain-Specific terms"


def test_clean_joins_lines_but_keeps_paragraphs():
    assert clean_text("first line\nsame sentence\n\nnew paragraph") == "first line same sentence\n\nnew paragraph"


def test_split_sizes_and_overlap():
    chunks = split_words(_words(300), size=150, overlap=30)
    assert len(chunks) == 3
    assert all(len(c.split()) <= 150 for c in chunks)
    assert chunks[0].split()[-30:] == chunks[1].split()[:30]  # overlap is shared


def test_short_text_skipped():
    assert split_words("page 3") == []


def test_chunks_keep_source_info():
    chunks = chunk_paper(PAPER, [{"page": 4, "text": _words(100)}])
    c = chunks[0]
    assert c["chunk_id"] == "2511.22858v1_p4_c0"
    assert c["page"] == 4 and c["title"] == "RAG Survey" and c["authors"] == "A. One, B. Two"


def test_references_section_dropped():
    pages = [{"page": 1, "text": _words(60, "body")},
             {"page": 2, "text": _words(60, "end") + "\nReferences\n" + _words(60, "cite")},
             {"page": 3, "text": _words(60, "cite")}]
    text = " ".join(c["text"] for c in chunk_paper(PAPER, pages))
    assert "body0" in text and "end0" in text
    assert "cite" not in text
