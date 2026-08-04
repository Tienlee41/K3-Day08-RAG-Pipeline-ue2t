"""Task 6 - lexical retrieval with BM25.

The index is built from the standardized Markdown corpus rather than from the
vector store.  This keeps lexical retrieval independent from Chroma and makes
it useful as the sparse half of the hybrid retriever.

``rank-bm25`` is used when it is installed (it is listed in
``requirements.txt``).  A small compatible implementation is kept as a
fallback so the module remains usable in a fresh/offline checkout while the
optional dependency is being installed.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Tokenize English/Vietnamese text without throwing punctuation away."""

    return _TOKEN_RE.findall(str(text).casefold())


def _load_corpus() -> list[dict]:
    """Load standardized Markdown files with stable metadata."""

    if not STANDARDIZED_DIR.exists():
        return []

    corpus: list[dict] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if not md_file.is_file():
            continue
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if relative_path.parts else "unknown"
        corpus.append(
            {
                "content": content,
                "metadata": {
                    "source": relative_path.as_posix(),
                    "filename": md_file.name,
                    "type": doc_type,
                },
            }
        )
    return corpus


# Public for the lab/demo and for callers that want to provide a custom corpus.
CORPUS: list[dict] = _load_corpus()
_BM25_INDEX: Any | None = None
_INDEX_CORPUS_ID: int | None = None


class _SimpleBM25:
    """Small BM25Okapi-compatible fallback.

    The formula follows BM25Okapi's defaults: ``k1=1.5`` and ``b=0.75``.
    It intentionally exposes only ``get_scores``, which is all this task
    needs.
    """

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(tokenized_corpus)
        self.doc_len = [len(document) for document in tokenized_corpus]
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size else 0.0
        self.term_frequency: list[dict[str, int]] = []
        document_frequency: dict[str, int] = {}

        for document in tokenized_corpus:
            frequencies: dict[str, int] = {}
            for token in document:
                frequencies[token] = frequencies.get(token, 0) + 1
            self.term_frequency.append(frequencies)
            for token in frequencies:
                document_frequency[token] = document_frequency.get(token, 0) + 1

        # The +1 form is the non-negative BM25Okapi IDF variant.  It prevents
        # common terms from producing negative scores in a tiny teaching corpus.
        self.idf = {
            token: math.log(1.0 + (self.corpus_size - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        if not self.corpus_size or not query_tokens:
            return [0.0] * self.corpus_size

        scores: list[float] = []
        for frequencies, document_length in zip(self.term_frequency, self.doc_len):
            score = 0.0
            length_factor = (
                self.k1 * (1.0 - self.b + self.b * document_length / self.avgdl)
                if self.avgdl
                else self.k1
            )
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                numerator = frequency * (self.k1 + 1.0)
                score += self.idf.get(token, 0.0) * numerator / (frequency + length_factor)
            scores.append(score)
        return scores


def build_bm25_index(corpus: list[dict]):
    """Build and cache a BM25 index for ``corpus``.

    The return value is a ``rank_bm25.BM25Okapi`` instance when the optional
    package is available, otherwise a local compatible implementation.  The
    cache is also updated so a caller can do ``build_bm25_index(custom_docs)``
    and immediately call :func:`lexical_search`.
    """

    global _BM25_INDEX, _INDEX_CORPUS_ID

    valid_corpus = [
        document
        for document in corpus
        if isinstance(document, dict) and str(document.get("content", "")).strip()
    ]
    tokenized_corpus = [_tokenize(document["content"]) for document in valid_corpus]

    if tokenized_corpus:
        try:
            from rank_bm25 import BM25Okapi

            index = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
        except ImportError:
            index = _SimpleBM25(tokenized_corpus)
    else:
        index = _SimpleBM25([])

    # Keep the public corpus aligned with the index, including custom corpora.
    CORPUS[:] = valid_corpus
    _BM25_INDEX = index
    _INDEX_CORPUS_ID = id(CORPUS)
    return index


def _get_index():
    global _BM25_INDEX, _INDEX_CORPUS_ID

    # Rebuild if a caller replaced or mutated the public corpus.  The length
    # check catches the common ``CORPUS.append(...)`` case; identity keeps the
    # cache tied to this module's public list.
    if _BM25_INDEX is None or _INDEX_CORPUS_ID != id(CORPUS):
        return build_bm25_index(CORPUS)
    expected_size = getattr(_BM25_INDEX, "corpus_size", None)
    if expected_size is not None and expected_size != len(CORPUS):
        return build_bm25_index(CORPUS)
    return _BM25_INDEX


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Return up to ``top_k`` positive BM25 matches sorted descending.

    Each result has the common retrieval schema::

        {"content": str, "score": float, "metadata": dict}

    Empty/invalid queries and non-positive ``top_k`` return an empty list.
    Documents with a zero score are omitted because they contain no query
    term and are not useful retrieval candidates.
    """

    if not isinstance(query, str) or not query.strip() or not isinstance(top_k, int) or top_k <= 0:
        return []
    if not CORPUS:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    index = _get_index()
    scores = index.get_scores(query_tokens)
    ranked_indices = sorted(
        range(min(len(CORPUS), len(scores))),
        key=lambda index: (-float(scores[index]), index),
    )

    results: list[dict] = []
    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0.0:
            continue
        document = CORPUS[index]
        results.append(
            {
                "content": str(document.get("content", "")),
                "score": score,
                "metadata": dict(document.get("metadata", {}) or {}),
            }
        )
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    for result in lexical_search("tuition fee payment methods", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
