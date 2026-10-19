import streamlit as st

from src.discovery.discover import discover
from src.gaps.compare import compare_papers
from src.gaps.coverage import coverage_report, papers_to_add
from src.gaps.disagree import MAX_PAPERS_PER_THEME, MAX_THEMES, choose_themes, find_disagreements
from src.gaps.extract import extract_aspects
from src.pipeline import index_papers, index_web_pages
from src.ranking.baseline import rank_papers
from src.ranking.judge import combined_score, judge_papers
from src.rag.agent import research_answer
from src.retrieval.store import get_collection
from src.web.search import search_web

st.set_page_config(page_title="ResearchLens AI", page_icon="🔎", layout="wide")


@st.cache_resource
def collection():
    return get_collection()  # opened once, shared across reruns


def pdf_link(source: dict) -> str:
    return f"{source['pdf_url']}#page={source['page']}" if source.get("pdf_url") else source.get("url", "")


def cite(source: dict) -> str:
    """'📄 [Title](pdf#page=4), p. 4' for papers, '🌐 [Title](url), section 2' for web pages."""
    if source.get("kind") == "web":
        return f"🌐 [{source['title']}]({source['url']}), section {source['page']}"
    return f"📄 [{source['title']}]({pdf_link(source)}), p. {source['page']}"


def show_answer(result: dict) -> None:
    st.markdown(result["answer"])
    if result["warning"]:
        st.warning(result["warning"])
    for n in result["cited"]:
        s = result["sources"][n - 1]
        st.caption(f"[{n}] {cite(s)}")
    if result["sources"]:
        with st.expander("All retrieved passages"):
            for i, s in enumerate(result["sources"], start=1):
                st.markdown(f"**[{i}]** {cite(s)} · distance {s['distance']:.2f}")
                st.text(s["text"])


def show_research(result: dict, key: int) -> None:
    st.markdown("##### 📄 From your papers")
    show_answer(result["grounded"])

    if result["general"]:
        st.markdown("##### 🧠 Beyond your papers")
        st.caption("From the model's general knowledge, not from your papers. It may be wrong or outdated.")
        st.markdown(result["general"]["text"])
        if result["general"]["warning"]:
            st.warning(result["general"]["warning"])

    if result["papers"] or result["new_papers"] or result.get("web"):
        st.markdown("##### 📚 Best papers for this question")
        for p in result["papers"][:3]:
            st.markdown(f"- In your collection: {cite(p)}")
        for p in result["new_papers"]:
            st.markdown(f"- New on arXiv: [{p['title']}]({p['url']}) · similarity {p['similarity']}")
        if result["new_papers"] and st.button(f"Add {len(result['new_papers'])} new papers to my collection",
                                              key=f"add_new_{key}"):
            with st.spinner("Downloading and indexing..."):
                n_chunks, errors = index_papers(result["new_papers"], collection())
            st.session_state.errors = st.session_state.get("errors", []) + errors
            st.session_state.flash_chat = f"Added {n_chunks} chunks. Ask again to use the new papers."
            st.rerun()
        for w in result.get("web", []):
            st.markdown(f"- 🌐 On the web: [{w['title']}]({w['url']}) · found via {w['found_via']}")
        if result.get("web") and st.button(f"Add {len(result['web'])} web articles to my collection",
                                           key=f"add_web_{key}"):
            with st.spinner("Reading web pages..."):
                pages, n_chunks, errors = index_web_pages(result["web"], collection())
            st.session_state.errors = st.session_state.get("errors", []) + errors
            st.session_state.flash_chat = (f"Added {len(pages)} web articles ({n_chunks} chunks). "
                                           f"{len(errors)} skipped (see sidebar). Ask again to use them.")
            st.rerun()
    for err in result["errors"]:
        st.error(err)


# ---------- Sidebar: ingestion phase ----------
with st.sidebar:
    st.header("1. Build a collection")
    topic = st.text_input("Research topic", placeholder="RAG evaluation techniques for agentic AI systems")
    max_pdfs = st.slider("Max PDFs to download", 1, 25, 8,
                         help="arXiv asks for 3 s between downloads, so more PDFs take longer.")

    use_web = st.checkbox("Also add web articles", value=True,
                          help="Searches the web (Tavily if TAVILY_API_KEY is set, otherwise Wikipedia + "
                               "Hacker News) and reads up to 3 pages that allow it (robots.txt).")
    use_judge = st.checkbox("LLM second opinion (experimental, slower)",
                            help="Ollama rates the selected papers 1-5 from their full text. "
                                 "Small models tend to rate everything highly, so the baseline score stays primary.")

    if st.button("Find and index papers", type="primary", disabled=not topic):
        with st.status("Working...", expanded=True) as status:
            st.write("Searching arXiv and Crossref...")
            papers, errors = discover(topic)
            st.write(f"Found {len(papers)} papers. Ranking...")
            papers = rank_papers(topic, papers)
            with_pdf = [p for p in papers if p.get("pdf_url")][:max_pdfs]  # best N, not first N
            st.write(f"Downloading and indexing the top {len(with_pdf)} open PDFs...")
            n_chunks, index_errors = index_papers(with_pdf, collection())
            if use_web:
                st.write("Searching the web and reading articles...")
                results, web_errors = search_web(topic, max_results=5)
                pages, n_web, fetch_errors = index_web_pages(results[:3], collection())
                st.session_state.web_pages = pages
                index_errors += web_errors + fetch_errors
                n_chunks += n_web
            if use_judge:
                st.write("Asking the LLM judge for a second opinion...")
                judged = {p["title"]: p for p in judge_papers(topic, with_pdf, collection())}
                papers = [judged.get(p["title"], p) for p in papers]
            st.session_state.papers = papers
            st.session_state.selected = with_pdf  # the papers gap analysis works on
            st.session_state.topic = topic
            st.session_state.pop("gaps", None)
            st.session_state.errors = errors + index_errors
            n_web_pages = len(st.session_state.get("web_pages", [])) if use_web else 0
            status.update(label=f"Indexed {n_chunks} chunks from {len(with_pdf)} PDFs and {n_web_pages} web pages",
                          state="complete")

    for err in st.session_state.get("errors", []):
        st.error(err)
    st.caption(f"Chunks in collection: {collection().count()}")

    with st.form("add_url", clear_on_submit=True):
        url = st.text_input("Add a web page (article, blog, docs)", placeholder="https://...")
        if st.form_submit_button("Add page") and url:
            with st.spinner("Reading the page..."):
                pages, n_chunks, errors = index_web_pages([{"url": url.strip(), "found_via": "link"}], collection())
            if pages:
                st.session_state.web_pages = st.session_state.get("web_pages", []) + pages
                st.success(f"Added \"{pages[0]['title']}\" ({n_chunks} chunks)")
            for err in errors:
                st.error(err)

    st.header("2. Answer settings")
    use_general = st.toggle("Also answer from general knowledge", value=True,
                            help="Adds a clearly labelled answer from the model's own knowledge. Turn off for papers-only answers.")
    search_new = st.toggle("Suggest new papers from arXiv", value=True,
                           help="Searches arXiv for each question (adds a few seconds).")
    search_web_toggle = st.toggle("Suggest web articles", value=True,
                                  help="Searches the web for each question (Tavily or Wikipedia + Hacker News).")


# ---------- Main: query phase ----------
st.title("🔎 ResearchLens AI")
st.caption("Each answer shows what your papers say (with citations) separately from the model's general knowledge.")

if st.session_state.get("papers"):
    with st.expander(f"Ranked papers ({len(st.session_state.papers)})"):
        st.caption("Score = 0.60 × relevance + 0.20 × recency + 0.15 × open PDF + 0.05 × has abstract. "
                   "Relevance is the cosine similarity between the topic and the paper's title + abstract. "
                   "The LLM score, if used, is a second opinion.")
        st.dataframe([{
            "score": p["score"],
            "relevance": round(p["score_parts"]["relevance"], 2),
            "similarity": p.get("similarity"),
            "recency": round(p["score_parts"]["recency"], 2),
            "PDF": "yes" if p.get("pdf_url") else "no",
            "LLM (1-5)": p.get("llm_score"),
            "combined": combined_score(p) if "llm_score" in p else None,
            "evidence": p.get("evidence_type", ""),
            "LLM reason": p.get("llm_reason", ""),
            "title": p["title"],
            "link": p.get("url", ""),
        } for p in st.session_state.papers], column_config={"link": st.column_config.LinkColumn()},
            hide_index=True)

STATUS_LABELS = {
    "covered": "✅ covered (2+ collected papers)",
    "thin": "⚠️ thin (1 collected paper)",
    "collection gap": "📚 collection gap (on-topic papers exist on arXiv)",
    "possibly underexplored": "❓ possibly underexplored (verify manually)",
}


def run_gap_analysis(topic: str, selected: list[dict]) -> dict:
    with st.status("Analysing gaps...", expanded=True) as status:
        extractions = {}
        for p in selected:
            st.write(f"Reading: {p['title'][:70]}")
            extractions[p["title"]] = extract_aspects(collection(), p)
        st.write("Checking which sub-topics the collection covers (includes a quick arXiv search)...")
        rows = coverage_report(collection(), topic, selected)
        status.update(label="Gap analysis complete", state="complete")
    return {"extractions": extractions, "comparison": compare_papers(extractions), "coverage": rows}


def show_gaps(gaps: dict, selected: list[dict]) -> None:
    n = len(selected)
    st.info(f"These findings describe the **{n} collected papers only**, not the whole research field. "
            "\"Possibly underexplored\" means neither the collection nor a quick arXiv search found it; "
            "it is a lead to verify, not proof that nobody has studied it.")

    st.subheader("What each paper reports")
    st.caption("Phrases are copied from the papers and checked against their text; invented items are dropped.")
    fmt = lambda items: "; ".join(f"{i['text']} (p. {i['page']})" for i in items)
    st.dataframe([{
        "paper": title,
        "methods": fmt(ex.get("methods", [])), "datasets": fmt(ex.get("datasets", [])),
        "metrics": fmt(ex.get("metrics", [])), "limitations": fmt(ex.get("limitations", [])),
        "dropped (unsupported)": ex.get("dropped", 0), "error": ex.get("error", ""),
    } for title, ex in gaps["extractions"].items()], hide_index=True)

    comparison = gaps["comparison"]
    for aspect, entries in comparison["shared"].items():
        if entries:
            st.markdown(f"**Shared {aspect}:** " + ", ".join(f"{e['label']} ({len(e['papers'])} papers)" for e in entries))
    for aspect, titles in comparison["missing"].items():
        if titles:
            st.markdown(f"**No {aspect} found in:** " + "; ".join(t[:60] for t in titles))

    st.subheader("Coverage of expected sub-topics")
    for row in gaps["coverage"]:
        with st.expander(f"{STATUS_LABELS[row['status']]}: {row['subtopic']}"):
            if row["papers"]:
                st.markdown("Mentioned in: " + "; ".join(f"{t} (p. {pg})" for t, pg in row["papers"].items()))
                st.text(row["evidence"]["passage"])
            for p in row["external"]:
                st.markdown(f"- arXiv: [{p['title']}]({p['url']}) · topic similarity {p['similarity']}")

    st.subheader("Possible disagreements")
    themes = choose_themes(gaps["coverage"], gaps["comparison"])
    if "disagreements" not in gaps:
        pairs = MAX_THEMES * MAX_PAPERS_PER_THEME
        if themes and st.button(f"Look for disagreements on {len(themes)} themes (slow: up to {pairs} LLM checks)"):
            with st.spinner("Comparing passages from different papers..."):
                gaps["disagreements"] = find_disagreements(collection(), selected, themes)
        elif not themes:
            st.caption("No theme is covered by 2+ papers yet, so there is nothing to compare.")
    if "disagreements" in gaps:
        found, checked = gaps["disagreements"]
        st.caption("Flagged only when the model says the passages conflict AND both quotes are found in the papers. "
                   "Read both passages: different settings or tasks can look like disagreement.")
        if not found:
            st.write(f"No disagreements found in {checked} pairs checked. "
                     "This does not mean the papers agree; the check is limited to a few passages per theme.")
        for d in found:
            st.markdown(f"**{d['theme']}**: {d['explanation']}")
            left, right = st.columns(2)
            for col, side in ((left, d["a"]), (right, d["b"])):
                col.markdown(f"*{side['title']}*, p. {side['page']}")
                col.markdown(f"> {side['claim']}")
                with col.expander("Full passage"):
                    st.text(side["text"])

    to_add = papers_to_add(gaps["coverage"], existing=st.session_state.get("papers", []))
    if to_add and st.button(f"Add {len(to_add)} suggested papers to the collection"):
        with st.spinner("Downloading and indexing..."):
            n_chunks, errors = index_papers(to_add, collection())
        st.session_state.selected = selected + to_add
        # re-rank so the new papers get scores too (and appear in the ranked table)
        st.session_state.papers = rank_papers(st.session_state.topic, st.session_state.get("papers", []) + to_add)
        st.session_state.errors = st.session_state.get("errors", []) + errors
        st.session_state.pop("gaps", None)
        st.session_state.flash = f"Added {n_chunks} chunks. Run the gap analysis again to see what changed."
        st.rerun()  # redraw everything, including the sidebar chunk count


if st.session_state.get("web_pages"):
    with st.expander(f"Web articles ({len(st.session_state.web_pages)})"):
        st.caption("Web pages are not peer-reviewed; weigh them below papers.")
        for w in st.session_state.web_pages:
            st.markdown(f"- 🌐 [{w['title']}]({w['url']}) · found via {w.get('found_via', 'link')}")

if st.session_state.get("selected"):
    with st.expander("Research gaps", expanded="gaps" in st.session_state or "flash" in st.session_state):
        if "flash" in st.session_state:
            st.success(st.session_state.pop("flash"))  # message saved before the rerun
        if st.button("Analyse gaps in the collected papers"):
            st.session_state.gaps = run_gap_analysis(st.session_state.topic, st.session_state.selected)
        if "gaps" in st.session_state:
            show_gaps(st.session_state.gaps, st.session_state.selected)

if "chat" not in st.session_state:
    st.session_state.chat = []

if "flash_chat" in st.session_state:
    st.success(st.session_state.pop("flash_chat"))

for i, msg in enumerate(st.session_state.chat):
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            show_research(msg["content"], key=i)

if question := st.chat_input("Ask anything about your research topic"):
    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if collection().count() == 0 and not use_general:
            st.info("No papers indexed yet. Enter a topic in the sidebar, or turn on general knowledge.")
        else:
            with st.spinner("Reading your papers, thinking, and looking for more papers..."):
                try:
                    result = research_answer(collection(), question, use_general, search_new, search_web_toggle)
                except Exception as e:  # e.g. Ollama not running
                    st.error(f"Could not answer: {e}. Is `ollama serve` running?")
                    st.stop()
            st.session_state.chat.append({"role": "assistant", "content": result})
            show_research(result, key=len(st.session_state.chat) - 1)
