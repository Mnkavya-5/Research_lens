import chromadb
from chromadb.config import Settings

from src.retrieval import embeddings

DB_DIR = "data/chroma"
COLLECTION = "chunks"


def get_collection(path: str = DB_DIR):
    client = chromadb.PersistentClient(  # saved to disk, survives restarts
        path=path, settings=Settings(anonymized_telemetry=False)  # keep everything local
    )
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def add_chunks(collection, chunks: list[dict], batch_size: int = 64) -> None:
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.upsert(  # upsert = insert, or replace if the id exists; safe to re-run
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=embeddings.embed([c["text"] for c in batch]),
            metadatas=[{k: v for k, v in c.items() if k not in ("chunk_id", "text")} for c in batch],
        )


def search(collection, question: str, k: int = 5) -> list[dict]:
    result = collection.query(query_embeddings=embeddings.embed([question]), n_results=k)
    return [
        {"chunk_id": cid, "text": doc, "distance": dist, **meta}
        for cid, doc, dist, meta in zip(result["ids"][0], result["documents"][0],
                                         result["distances"][0], result["metadatas"][0])
    ]
