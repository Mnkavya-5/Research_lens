import re

from src.evaluation.metrics import sentence_support
from src.llm.ollama_client import ask_ollama
from src.retrieval.store import search

NOT_FOUND = "The collected papers do not contain enough information to answer this."

PROMPT = """Sources:
{sources}

Question: {question}

Instructions: Answer the question using only the sources above. Write 2-4 short sentences.
End every sentence with the number of the source it comes from, in square brackets, e.g. [1] or [3].
If no source is relevant to the question, reply only: "{not_found}"
Answer:"""


def location(chunk: dict) -> str:
    return f"web article, section {chunk['page']}" if chunk.get("kind") == "web" else f"p. {chunk['page']}"


def build_prompt(question: str, chunks: list[dict]) -> str:
    sources = "\n\n".join(
        f"[{i}] {c['title']} ({location(c)})\n{c['text']}" for i, c in enumerate(chunks, start=1)
    )
    return PROMPT.format(not_found=NOT_FOUND, sources=sources, question=question)


def check_citations(answer: str, n_sources: int) -> tuple[list[int], list[int]]:
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    valid = [n for n in cited if 1 <= n <= n_sources]
    invalid = [n for n in cited if not 1 <= n <= n_sources]
    return valid, invalid


def answer_question(collection, question: str, k: int = 8) -> dict:  # k=8 beat k=5 in eval/run_eval.py
    chunks = search(collection, question, k)
    if not chunks:
        return {"answer": NOT_FOUND, "sources": [], "cited": [], "invalid_citations": [], "unsupported": [],
                "warning": "no papers indexed"}

    answer = ask_ollama(build_prompt(question, chunks), temperature=0).strip()
    cited, invalid = check_citations(answer, len(chunks))

    support = [] if NOT_FOUND in answer else sentence_support(answer, chunks)
    unsupported = [r["sentence"] for r in support if not r["supported"]]

    warning = None
    if invalid:
        warning = f"answer cites sources that do not exist: {invalid}"
    elif not cited and NOT_FOUND not in answer:
        warning = "answer has no citations; treat it as unverified"
    elif unsupported:
        warning = (f"{len(unsupported)} of {len(support)} sentences are not clearly supported by the passage "
                   "they cite; check them against the sources")
    return {"answer": answer, "sources": chunks, "cited": cited, "invalid_citations": invalid,
            "unsupported": unsupported, "warning": warning}
