"""Task 7 - reranking utilities.

The default implementation is Reciprocal Rank Fusion (RRF), which is
deterministic and needs no API key.  Optional cross-encoder reranking is
supported through Jina when ``JINA_API_KEY`` is configured; without a key it
falls back to a lightweight lexical relevance score.
"""

from __future__ import annotations

import math
import os
import re
from typing import Iterable


_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(text).casefold()))


def _safe_top_k(top_k: int) -> int:
    return top_k if isinstance(top_k, int) and top_k > 0 else 0


def _cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if len(left_values) != len(right_values) or not left_values:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left_values, right_values)) / (left_norm * right_norm)


def _local_relevance(query: str, content: str) -> float:
    """Cheap, deterministic relevance score used when no reranker API is set."""

    query_tokens = _tokens(query)
    content_tokens = _tokens(content)
    if not query_tokens or not content_tokens:
        return 0.0
    coverage = len(query_tokens & content_tokens) / len(query_tokens)
    phrase_bonus = 0.15 if query.casefold().strip() in content.casefold() else 0.0
    return min(1.0, coverage + phrase_bonus)


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """Rerank candidates with Jina when configured, otherwise locally.

    The local path is intentionally dependency-free.  It makes the public
    function useful in tests and demos without silently making a network call.
    """

    limit = _safe_top_k(top_k)
    if not limit or not isinstance(query, str) or not candidates:
        return []

    api_key = os.getenv("JINA_API_KEY", "").strip()
    if api_key:
        try:
            import requests

            response = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": os.getenv(
                        "JINA_RERANKER_MODEL", "jinaai/jina-reranker-v2-base-multilingual"
                    ),
                    "query": query,
                    "documents": [str(candidate.get("content", "")) for candidate in candidates],
                    "top_n": limit,
                },
                timeout=float(os.getenv("JINA_TIMEOUT", "20")),
            )
            response.raise_for_status()
            payload = response.json()
            api_results = payload.get("results", [])
            reranked: list[dict] = []
            for result in api_results:
                index = int(result.get("index", -1))
                if 0 <= index < len(candidates):
                    item = dict(candidates[index])
                    item["score"] = float(result.get("relevance_score", 0.0))
                    reranked.append(item)
            if reranked:
                return reranked[:limit]
        except (ImportError, ValueError, TypeError, OSError, RuntimeError, KeyError) as exc:
            # A transient optional service failure should not take down local
            # retrieval.  Continue with the deterministic fallback below.
            del exc
        except Exception:
            # requests can raise library-specific HTTP/JSON exceptions.  Keep
            # the same offline-safe behavior without coupling this module to it.
            pass

    scored: list[tuple[int, float, float]] = []
    original_scores = [float(candidate.get("score", 0.0) or 0.0) for candidate in candidates]
    max_original = max(original_scores, default=0.0)
    min_original = min(original_scores, default=0.0)
    original_span = max_original - min_original
    for index, candidate in enumerate(candidates):
        original = (
            (original_scores[index] - min_original) / original_span
            if original_span > 0
            else 0.0
        )
        relevance = _local_relevance(query, str(candidate.get("content", "")))
        # Relevance dominates; the incoming score is only a stable tie-breaker.
        scored.append((index, 0.8 * relevance + 0.2 * original, original))

    scored.sort(key=lambda item: (-item[1], -item[2], item[0]))
    results: list[dict] = []
    for index, score, _ in scored[:limit]:
        item = dict(candidates[index])
        item["score"] = float(score)
        results.append(item)
    return results


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """Select relevant and diverse candidates with Maximal Marginal Relevance."""

    limit = _safe_top_k(top_k)
    if not limit or not candidates:
        return []
    if not 0.0 <= lambda_param <= 1.0:
        raise ValueError("lambda_param must be between 0 and 1")

    selected_indices: list[int] = []
    remaining = set(range(len(candidates)))
    while remaining and len(selected_indices) < limit:
        best_index: int | None = None
        best_mmr = float("-inf")
        for index in sorted(remaining):
            embedding = candidates[index].get("embedding")
            if embedding is None:
                relevance = float(candidates[index].get("score", 0.0) or 0.0)
            else:
                relevance = _cosine_similarity(query_embedding, embedding)

            max_similarity = 0.0
            for selected_index in selected_indices:
                selected_embedding = candidates[selected_index].get("embedding")
                if embedding is not None and selected_embedding is not None:
                    max_similarity = max(
                        max_similarity,
                        _cosine_similarity(embedding, selected_embedding),
                    )
            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_similarity
            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_index = index

        if best_index is None:  # Defensive guard for malformed input.
            break
        selected_indices.append(best_index)
        remaining.remove(best_index)

    results: list[dict] = []
    for index in selected_indices:
        item = dict(candidates[index])
        embedding = item.get("embedding")
        relevance = (
            float(item.get("score", 0.0) or 0.0)
            if embedding is None
            else _cosine_similarity(query_embedding, embedding)
        )
        max_similarity = max(
            (
                _cosine_similarity(embedding, candidates[selected].get("embedding"))
                for selected in selected_indices
                if selected != index
                and embedding is not None
                and candidates[selected].get("embedding") is not None
            ),
            default=0.0,
        )
        item["score"] = float(lambda_param * relevance - (1.0 - lambda_param) * max_similarity)
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """Fuse ranked result lists with Reciprocal Rank Fusion.

    Ranks are one-based, so the first item in one list contributes
    ``1 / (k + 1)``.  Candidates with the same ``content`` are treated as the
    same document and their contributions are added across rankers.
    """

    limit = _safe_top_k(top_k)
    if not limit or not isinstance(k, int) or k < 0:
        return []

    scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}
    first_seen: dict[str, int] = {}
    seen_counter = 0

    for ranked_list in ranked_lists or []:
        if not ranked_list:
            continue
        for rank, candidate in enumerate(ranked_list, start=1):
            if not isinstance(candidate, dict):
                continue
            content = str(candidate.get("content", ""))
            if not content:
                continue
            if content not in first_seen:
                first_seen[content] = seen_counter
                seen_counter += 1
            scores[content] = scores.get(content, 0.0) + 1.0 / (k + rank)
            content_map[content] = dict(candidate)

    ordered_contents = sorted(
        scores,
        key=lambda content: (-scores[content], first_seen[content]),
    )
    results: list[dict] = []
    for content in ordered_contents[:limit]:
        item = dict(content_map[content])
        item["score"] = float(scores[content])
        results.append(item)
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",
) -> list[dict]:
    """Unified reranking interface.

    ``rrf`` normally receives multiple lists through :func:`rerank_rrf`.  For
    this single-list interface, the candidates are treated as one already
    ranked list; this preserves rank order and gives every result a proper RRF
    score.  Passing ``list[list[dict]]`` is also supported for convenience.
    """

    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        raise ValueError("method='mmr' requires query_embedding; call rerank_mmr directly")
    if method == "rrf":
        if candidates and isinstance(candidates[0], list):
            return rerank_rrf(candidates, top_k=top_k)
        return rerank_rrf([candidates], top_k=top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Tuition fee payment schedule", "score": 0.8, "metadata": {}},
        {"content": "Scholarship eligibility requirements", "score": 0.6, "metadata": {}},
        {"content": "Library study room booking guide", "score": 0.5, "metadata": {}},
    ]
    for result in rerank("tuition fee payment", dummy_candidates, top_k=2):
        print(f"[{result['score']:.3f}] {result['content']}")
