from src.discovery.dedupe import dedupe_papers, normalize_title


def test_lowercase():
    assert normalize_title("RAG Evaluation") == "rag evaluation"


def test_punctuation_removed():
    assert normalize_title("RAG: a survey.") == "rag a survey"


def test_extra_spaces():
    assert normalize_title("  RAG   for   fintech ") == "rag for fintech"


def test_duplicates_merged_keeping_pdf():
    arxiv = {"title": "RAG for Fintech: Agentic Design", "abstract": "A",
             "pdf_url": "https://arxiv.org/pdf/1", "source": "arxiv"}
    crossref = {"title": "RAG for fintech - agentic design.", "abstract": None,
                "pdf_url": None, "source": "crossref"}
    result = dedupe_papers([crossref, arxiv])
    assert len(result) == 1
    assert result[0]["pdf_url"] == "https://arxiv.org/pdf/1"


def test_different_papers_kept():
    a = {"title": "RAG evaluation", "pdf_url": None, "abstract": None}
    b = {"title": "RAG security", "pdf_url": None, "abstract": None}
    assert len(dedupe_papers([a, b])) == 2
