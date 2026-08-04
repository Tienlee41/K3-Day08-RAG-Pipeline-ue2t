"""Task 5 - Dense semantic retrieval over the Task 4 Chroma collection."""

from __future__ import annotations

try:
    from .task4_chunking_indexing import get_collection, get_embedding_model
except ImportError:  # Supports `python src/task5_semantic_search.py` too.
    from task4_chunking_indexing import get_collection, get_embedding_model


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Return the most semantically similar indexed chunks.

    The query is embedded with the exact model used during indexing.  Chroma
    returns cosine distance; because the collection is configured with
    ``hnsw:space=cosine``, ``1 - distance`` is the cosine similarity used by
    the rest of the RAG pipeline.  Results are explicitly sorted descending
    and clipped to ``[0, 1]`` for a meaningful relevance score/threshold.
    """

    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []

    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    model = get_embedding_model()
    query_vector = model.encode(
        [query.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0].astype(float).tolist()
    raw = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    results: list[dict] = []
    for content, metadata, distance in zip(documents, metadatas, distances):
        # Cosine distance may be slightly outside its theoretical range due
        # to floating-point arithmetic; clamp the public score defensively.
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        results.append(
            {
                "content": str(content),
                "score": round(score, 6),
                "metadata": dict(metadata or {}),
            }
        )

    results.sort(key=lambda result: result["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in semantic_search("what is the tuition fee", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
