import re
import zlib

import pytest

from src.rag import answer as rag
from src.retrieval import embeddings, store

DIM = 64


def fake_embed(texts):
    # Bag-of-words hashing: texts sharing words get similar vectors. Fast, no model download.
    vectors = []
    for text in texts:
        v = [0.0] * DIM
        for word in re.findall(r"\w+", text.lower()):  # ignore punctuation ("faithfulness?" = "faithfulness")
            v[zlib.crc32(word.encode()) % DIM] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        vectors.append([x / norm for x in v])
    return vectors


def _chunk(cid, text, page=1):
    return {"chunk_id": cid, "text": text, "page": page, "title": f"Paper {cid}",
            "authors": "A. One", "published": "2025", "url": "u", "pdf_url": "p"}


@pytest.fixture
def collection(monkeypatch, tmp_path):
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    col = store.get_collection(str(tmp_path / "chroma"))
    store.add_chunks(col, [
        _chunk("a", "faithfulness metrics measure whether answers match retrieved evidence", page=3),
        _chunk("b", "fintech knowledge base acronyms complicate retrieval"),
        _chunk("c", "legal norms constrain litigation support systems"),
    ])
    return col


def test_search_returns_closest_chunk_with_metadata(collection):
    hits = store.search(collection, "faithfulness metrics for answers", k=2)
    assert hits[0]["chunk_id"] == "a"
    assert hits[0]["page"] == 3 and hits[0]["title"] == "Paper a"


def test_upsert_does_not_duplicate(collection):
    store.add_chunks(collection, [_chunk("a", "faithfulness metrics updated text")])
    assert collection.count() == 3


def test_prompt_numbers_sources():
    prompt = rag.build_prompt("q?", [_chunk("a", "text one", 3), _chunk("b", "text two")])
    assert "[1] Paper a (p. 3)" in prompt and "[2] Paper b (p. 1)" in prompt


def test_check_citations_flags_invented_sources():
    assert rag.check_citations("Claim [1]. Other [2][7].", n_sources=5) == ([1, 2], [7])


def test_answer_warns_when_no_citations(monkeypatch, collection):
    monkeypatch.setattr(rag, "ask_ollama", lambda prompt, **kw: "Faithfulness is important.")
    result = rag.answer_question(collection, "what is faithfulness?", k=2)
    assert result["warning"] == "answer has no citations; treat it as unverified"


def test_answer_with_valid_citations(monkeypatch, collection):
    monkeypatch.setattr(rag, "ask_ollama", lambda prompt, **kw: "Faithfulness metrics measure whether answers match evidence [1].")
    result = rag.answer_question(collection, "what is faithfulness?", k=2)
    assert result["cited"] == [1] and result["warning"] is None


def test_answer_warns_on_unsupported_sentence(monkeypatch, collection):
    # regression from evaluation: "The boiling point of mercury is 253 K [1]." passed the citation check
    monkeypatch.setattr(rag, "ask_ollama", lambda prompt, **kw: "The boiling point of mercury is reported as 253 K [1].")
    result = rag.answer_question(collection, "boiling point of mercury?", k=2)
    assert result["unsupported"] and "not clearly supported" in result["warning"]
