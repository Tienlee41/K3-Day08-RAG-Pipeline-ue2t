"""Task 10 - citation-aware generation over retrieved evidence.

The LLM call is optional.  With an OpenRouter/OpenAI key the module uses the
OpenAI-compatible chat API; without a key it returns a deterministic
extractive answer with source citations.  This keeps local demos and tests
useful while preserving the same retrieval/context contract used in
production.
"""

from __future__ import annotations

import os
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv()

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:  # Supports ``python src/task10_generation.py`` as well.
    from task9_retrieval_pipeline import retrieve


TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")


SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về dịch vụ và chính sách đại học.

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin trong CONTEXT; không bịa đặt hoặc suy luận vượt quá nguồn.
2. Mỗi khẳng định thực tế phải có citation ngay sau câu, dùng đúng nhãn Source trong context,
   ví dụ: [tuition-fees-rmit.md].
3. Nếu context không đủ bằng chứng, hãy nói: "Tôi không thể xác minh thông tin này từ các nguồn hiện có."
4. Trả lời ngắn gọn, rõ ràng bằng tiếng Việt.
"""

_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Reorder ranked chunks as ``front + reversed(back)``.

    For ``[1, 2, 3, 4, 5]`` this returns ``[1, 3, 5, 4, 2]``.  The most
    relevant chunk stays first, while the next most relevant chunks are moved
    to the beginning/end positions that language models tend to attend to.
    The input list is never mutated.
    """

    if not isinstance(chunks, list) or len(chunks) <= 2:
        return list(chunks) if isinstance(chunks, list) else []
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def _citation_label(chunk: dict, index: int = 0) -> str:
    metadata = chunk.get("metadata", {}) or {}
    source = metadata.get("source") or metadata.get("filename")
    if source:
        return str(source)
    section = metadata.get("section")
    if section:
        return str(section)
    return f"Source {index + 1}"


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks with stable source labels for citations."""

    if not chunks:
        return ""

    context_parts: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        metadata = dict(chunk.get("metadata", {}) or {})
        source = _citation_label(chunk, index - 1)
        doc_type = metadata.get("type", "unknown")
        section = metadata.get("section")
        page = metadata.get("page_index", metadata.get("page"))
        score = chunk.get("score")

        labels = [f"Document {index}", f"Source: {source}", f"Type: {doc_type}"]
        if section:
            labels.append(f"Section: {section}")
        if page is not None:
            labels.append(f"Page: {page}")
        if isinstance(score, (int, float)):
            labels.append(f"Score: {float(score):.4f}")

        content = str(chunk.get("content", "")).strip()
        if not content:
            continue
        context_parts.append(f"[{ ' | '.join(labels) }]\n{content}")
    return "\n\n---\n\n".join(context_parts)


def _query_tokens(query: str) -> set[str]:
    return set(_TOKEN_RE.findall(query.casefold()))


def _extractive_answer(query: str, chunks: list[dict]) -> str:
    """Produce a citation-bearing answer without an external LLM."""

    if not chunks:
        return "Tôi không thể xác minh thông tin này từ các nguồn hiện có."

    query_tokens = _query_tokens(query)
    evidence: list[tuple[float, int, str, str]] = []
    for chunk_index, chunk in enumerate(chunks):
        content = str(chunk.get("content", "")).strip()
        if not content:
            continue
        sentences = [part.strip() for part in _SENTENCE_RE.split(content) if part.strip()]
        if not sentences:
            sentences = [content]
        for sentence_index, sentence in enumerate(sentences):
            sentence_tokens = _query_tokens(sentence)
            overlap = (
                len(query_tokens & sentence_tokens) / len(query_tokens)
                if query_tokens
                else 0.0
            )
            phrase_bonus = 0.15 if query.casefold().strip() in sentence.casefold() else 0.0
            evidence.append(
                (
                    min(1.0, overlap + phrase_bonus),
                    -sentence_index,
                    _citation_label(chunk, chunk_index),
                    sentence,
                )
            )

    evidence.sort(key=lambda item: (-item[0], item[1]))
    selected: list[tuple[str, str]] = []
    seen_sentences: set[str] = set()
    for score, _, label, sentence in evidence:
        if sentence.casefold() in seen_sentences:
            continue
        seen_sentences.add(sentence.casefold())
        # With no lexical overlap, use the first evidence sentence rather than
        # inventing a connection between the query and unrelated content.
        if score > 0.0 or not selected:
            selected.append((sentence[:600], label))
        if len(selected) >= 2:
            break

    if not selected:
        return "Tôi không thể xác minh thông tin này từ các nguồn hiện có."
    bullets = [f"- {sentence} [{label}]" for sentence, label in selected]
    return "Dựa trên các tài liệu đã truy xuất:\n" + "\n".join(bullets)


def _call_llm(query: str, context: str) -> str | None:
    """Call an OpenAI-compatible endpoint when credentials are available."""

    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    api_key = openrouter_key or openai_key
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if openrouter_key:
            client_kwargs["base_url"] = os.getenv(
                "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
            )
            extra_headers = {}
            if os.getenv("OPENROUTER_HTTP_REFERER"):
                extra_headers["HTTP-Referer"] = os.environ["OPENROUTER_HTTP_REFERER"]
            if os.getenv("OPENROUTER_APP_TITLE"):
                extra_headers["X-Title"] = os.environ["OPENROUTER_APP_TITLE"]
            if extra_headers:
                client_kwargs["default_headers"] = extra_headers
        elif os.getenv("OPENAI_BASE_URL"):
            client_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]

        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", LLM_MODEL),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context or '(empty)'}\n\nQUESTION:\n{query}",
                },
            ],
            temperature=float(os.getenv("LLM_TEMPERATURE", str(TEMPERATURE))),
            top_p=float(os.getenv("LLM_TOP_P", str(TOP_P))),
        )
        message = response.choices[0].message.content
        if isinstance(message, str) and message.strip():
            return message.strip()
    except Exception:
        # Generation remains useful offline and when a provider is temporarily
        # unavailable; the caller will use the extractive answer instead.
        return None
    return None


def _ensure_citation(answer: str, chunks: list[dict]) -> str:
    """Append source labels if an LLM returned an answer without citations."""

    if not chunks or re.search(r"\[[^\]]+\]", answer):
        return answer
    labels = list(dict.fromkeys(f"[{_citation_label(chunk, i)}]" for i, chunk in enumerate(chunks)))
    return f"{answer}\n\nNguồn: {', '.join(labels)}"


def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    context_chunks: list[dict] | None = None,
) -> dict:
    """Retrieve evidence, reorder it, and generate a citation-aware answer.

    ``context_chunks`` is optional for callers/tests that already performed
    retrieval.  Normally the function calls :func:`retrieve` itself.
    """

    if not isinstance(query, str) or not query.strip():
        return {
            "answer": "Tôi không thể xác minh thông tin này từ các nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    if context_chunks is None:
        limit = top_k if isinstance(top_k, int) and top_k > 0 else 0
        chunks = retrieve(query.strip(), top_k=limit) if limit else []
    else:
        chunks = [chunk for chunk in context_chunks if isinstance(chunk, dict)]
        if isinstance(top_k, int) and top_k > 0:
            chunks = chunks[:top_k]

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    answer = _call_llm(query.strip(), context)
    if not answer:
        answer = _extractive_answer(query.strip(), reordered)
    answer = _ensure_citation(answer, reordered)

    retrieval_source = "none"
    if reordered:
        sources = [str(chunk.get("source", "hybrid")) for chunk in reordered]
        retrieval_source = "pageindex" if all(source == "pageindex" for source in sources) else "hybrid"

    return {
        "answer": answer,
        "sources": reordered,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    for query in [
        "Học phí tại RMIT Vietnam là bao nhiêu?",
        "Làm sao để đặt phòng học nhóm ở thư viện?",
        "Sinh viên quốc tế có những học bổng nào?",
    ]:
        print(f"\n{'=' * 70}\nQ: {query}\n{'=' * 70}")
        result = generate_with_citation(query)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} | via {result['retrieval_source']}]")
