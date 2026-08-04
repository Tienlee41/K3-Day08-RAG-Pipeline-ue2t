"""Task 9 - hybrid retrieval pipeline with a PageIndex fallback.

The pipeline deliberately keeps the semantic score separate from the RRF
score.  RRF is a rank-fusion score and is useful for ordering candidates, but
it is not a calibrated relevance score suitable for the fallback threshold.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# This is the calibrated/default threshold for the current lab configuration.
# It is compared only with semantic_search's original cosine similarity.
SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "rrf"  # "cross_encoder" | "mmr" | "rrf"


def _safe_search(search_fn: Callable[[str, int], list[dict]], query: str, limit: int) -> list[dict]:
    """Run one retriever without allowing an optional backend to crash RAG."""

    try:
        results = search_fn(query, top_k=limit)
    except Exception:
        return []
    if not isinstance(results, list):
        return []
    return [result for result in results if isinstance(result, dict)]


def _normalise_result(result: dict, source: str) -> dict:
    """Ensure every branch of the pipeline returns the common result schema."""

    item = dict(result)
    item["content"] = str(item.get("content", ""))
    item["score"] = float(item.get("score", 0.0) or 0.0)
    item["metadata"] = dict(item.get("metadata", {}) or {})
    item["source"] = source
    return item


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Retrieve the best evidence using dense+sparse fusion and fallback.

    Steps:

    1. Run semantic and lexical retrieval concurrently.
    2. Fuse both ranked lists with RRF.
    3. Optionally apply the configured reranker.
    4. Compare the *original semantic cosine score* with ``score_threshold``.
    5. If semantic evidence is weak, return PageIndex results when available.
    """

    if not isinstance(query, str) or not query.strip():
        return []
    if not isinstance(top_k, int) or top_k <= 0:
        return []
    try:
        threshold = float(score_threshold)
    except (TypeError, ValueError):
        threshold = SCORE_THRESHOLD

    query = query.strip()
    candidate_limit = max(top_k * 2, top_k)

    # Keep the two retrieval calls independent: Chroma/model failures should
    # still leave lexical or PageIndex retrieval available.
    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(_safe_search, semantic_search, query, candidate_limit)
        sparse_future = executor.submit(_safe_search, lexical_search, query, candidate_limit)
        dense_results = dense_future.result()
        sparse_results = sparse_future.result()

    # Preserve this before RRF overwrites each candidate's public score.
    best_dense_score = max(
        (float(result.get("score", 0.0) or 0.0) for result in dense_results),
        default=0.0,
    )

    merged = rerank_rrf([dense_results, sparse_results], top_k=candidate_limit)
    merged = [_normalise_result(result, "hybrid") for result in merged]

    if use_reranking and merged:
        try:
            final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        except (ValueError, RuntimeError, TypeError):
            # RRF fusion itself is already a valid deterministic reranking
            # stage, so an optional secondary reranker may fail safely.
            final_results = merged[:top_k]
    else:
        final_results = merged[:top_k]
    final_results = [_normalise_result(result, "hybrid") for result in final_results]

    # Important: do not compare the RRF score (~0.016 with k=60) to the
    # threshold.  Only dense cosine similarity has the intended [0, 1] scale.
    if best_dense_score < threshold:
        fallback = _safe_search(pageindex_search, query, top_k)
        if fallback:
            return [_normalise_result(result, "pageindex") for result in fallback[:top_k]]

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "What is the tuition fee at RMIT Vietnam?",
        "How do I book a library study room?",
        "What scholarships are available for international students?",
        "xyzabc123nonsense",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 60)
        results = retrieve(query, top_k=3)
        for index, result in enumerate(results, 1):
            print(
                f"  {index}. [{result['score']:.3f}] [{result['source']}] "
                f"{result['content'][:80]}..."
            )
