from src.discovery.discover import discover
from src.processing.chunking import chunk_paper
from src.processing.download import download_all
from src.processing.pdf import ExtractionError, extract_pages
from src.retrieval.store import add_chunks, get_collection
from src.web.fetch import FetchError, chunk_page, fetch_page


def index_papers(papers: list[dict], collection) -> tuple[int, list[str]]:
    """Download, extract, chunk and store papers. Returns (chunks added, errors)."""
    downloaded, errors = download_all(papers)
    total = 0
    for paper in downloaded:
        try:
            chunks = chunk_paper(paper, extract_pages(paper["pdf_path"]))
        except ExtractionError as e:
            errors.append(str(e))
            continue
        add_chunks(collection, chunks)
        total += len(chunks)
    return total, errors


def index_web_pages(results: list[dict], collection) -> tuple[list[dict], int, list[str]]:
    """Fetch, chunk and store web pages. Returns (pages indexed, chunks added, errors)."""
    pages, total, errors = [], 0, []
    for r in results:
        try:
            page = fetch_page(r["url"])
        except FetchError as e:
            errors.append(str(e))
            continue
        chunks = chunk_page(page)
        add_chunks(collection, chunks)
        pages.append({**page, "found_via": r.get("found_via", "link")})
        total += len(chunks)
    return pages, total, errors


def ingest(topic: str, collection=None, max_pdfs: int = 10) -> dict:
    """The full ingestion phase: topic -> papers -> chunks in ChromaDB."""
    collection = collection or get_collection()
    papers, errors = discover(topic)
    with_pdf = [p for p in papers if p.get("pdf_url")][:max_pdfs]  # cap download time
    n_chunks, index_errors = index_papers(with_pdf, collection)
    return {"papers": papers, "chunks_added": n_chunks, "errors": errors + index_errors}
