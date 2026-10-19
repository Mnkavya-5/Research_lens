import re

import requests

from src.llm.ollama_client import ask_ollama

PROMPT = """You help search an academic paper database.
Topic: {topic}
Write {n} different short keyword search queries (2-5 words each) for this topic.
Output one query per line. No numbering, no bullets, no explanations."""


def _clean_lines(text: str) -> list[str]:
    queries = []
    for line in text.splitlines():
        line = re.sub(r"^[\s\-\*•\d\.\)]+", "", line)  # strip "1.", "-", "•" prefixes
        line = line.strip().strip('"').strip()
        if line and len(line.split()) <= 8:             # skip blanks and chatty sentences
            queries.append(line)
    return queries


def generate_queries(topic: str, n: int = 3) -> list[str]:
    try:
        reply = ask_ollama(PROMPT.format(topic=topic, n=n), temperature=0)  # same topic -> same queries
    except requests.RequestException:
        reply = ""  # Ollama down or slow: fall back to the topic alone

    queries = [topic]  # the user's own words always come first
    seen = {topic.lower()}
    for q in _clean_lines(reply)[:n]:
        if q.lower() not in seen:
            seen.add(q.lower())
            queries.append(q)
    return queries
