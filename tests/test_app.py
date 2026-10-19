from streamlit.testing.v1 import AppTest


def test_app_loads_without_errors():
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "🔎 ResearchLens AI"


def test_index_button_disabled_until_topic_entered():
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert at.sidebar.button[0].disabled
    at.sidebar.text_input[0].set_value("RAG evaluation").run()
    assert not at.sidebar.button[0].disabled


def test_ranked_table_survives_unscored_papers_being_added(monkeypatch):
    # regression: papers added from gap analysis had no "score" and crashed the ranked table
    at = AppTest.from_file("../app.py", default_timeout=60)
    at.session_state["papers"] = [{"title": "A", "score": 0.5, "pdf_url": "u",
                                   "score_parts": {"relevance": 1.0, "recency": 1.0, "open_pdf": 1.0, "has_abstract": 0.0}}]
    at.run()
    assert not at.exception
