import requests

from src.discovery import query_gen
from src.discovery.query_gen import _clean_lines, generate_queries


def test_clean_strips_numbering_and_bullets():
    reply = "1. RAG evaluation metrics\n- agentic retrieval\n• \"LLM agents benchmark\"\n"
    assert _clean_lines(reply) == ["RAG evaluation metrics", "agentic retrieval", "LLM agents benchmark"]


def test_clean_drops_chatty_sentences():
    reply = "Here are some queries that you could use for your search topic today:\nRAG faithfulness"
    assert _clean_lines(reply) == ["RAG faithfulness"]


def test_topic_always_first(monkeypatch):
    monkeypatch.setattr(query_gen, "ask_ollama", lambda prompt, **kw: "rag metrics\nagent evaluation")
    assert generate_queries("RAG for agents") == ["RAG for agents", "rag metrics", "agent evaluation"]


def test_fallback_when_ollama_down(monkeypatch):
    def broken(prompt, **kw):
        raise requests.ConnectionError("Ollama not running")
    monkeypatch.setattr(query_gen, "ask_ollama", broken)
    assert generate_queries("RAG for agents") == ["RAG for agents"]


def test_duplicate_of_topic_removed(monkeypatch):
    monkeypatch.setattr(query_gen, "ask_ollama", lambda prompt, **kw: "rag for agents\nrag metrics")
    assert generate_queries("RAG for agents") == ["RAG for agents", "rag metrics"]
