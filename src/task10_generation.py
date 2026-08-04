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
LLM_MAX_TOKENS = 512


SYSTEM_PROMPT = """Bạn là trợ lý trả lời câu hỏi về dịch vụ và chính sách đại học.

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin trong CONTEXT; không bịa đặt hoặc suy luận vượt quá nguồn.
2. Mỗi khẳng định thực tế phải có citation ngay sau câu, dùng đúng nhãn Source trong context,
   ví dụ: [tuition-fees-rmit.md].
3. Nếu context không đủ bằng chứng, hãy nói: "Tôi không thể xác minh thông tin này từ các nguồn hiện có."
4. Trả lời ngắn gọn, rõ ràng bằng tiếng Việt.
"""

_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?。！？])\s+|\n{2,}")
_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "the", "this", "to",
    "what", "when", "where", "which", "who", "why", "with",
    "bao", "bằng", "bị", "các", "cách", "cho", "có", "còn", "của", "đã",
    "đang", "đây", "để", "được", "đâu", "hai", "hay", "hãy", "khi", "là",
    "làm", "một", "nào", "này", "như", "những", "qua", "sao", "sẽ", "tại",
    "theo", "thế", "thì", "trong", "từ", "và", "về", "với", "xin", "ở",
}

# The corpus contains both Vietnamese and English pages. Add a few stable
# domain aliases so a Vietnamese question can match an English source.
_PHRASE_ALIASES = {
    "học phí": {"tuition", "fee", "fees"},
    "học bổng": {"scholarship", "scholarships"},
    "thanh toán": {"payment", "payments", "pay"},
    "phương thức thanh toán": {"payment", "payments", "method", "methods"},
    "thư viện": {"library"},
    "hỗ trợ": {"support"},
    "học tập": {"learning", "study", "academic"},
    "academic achievement": {"academic", "achievement", "scholarship", "scholarships"},
    "dịch vụ": {"service", "services"},
    "đăng ký": {"register", "registration", "enrol", "enrollment"},
    "học phần": {"course", "courses"},
    "chỗ ở": {"accommodation", "housing"},
    "ký túc": {"accommodation", "dormitory"},
}

_UNSUPPORTED_TOPIC_RULES = (
    # These topics are intentionally refused unless the retrieved evidence
    # contains the topic itself; otherwise a generic university word such as
    # "Hà Nội" or "hỗ trợ" can produce a misleading answer.
    (("thời tiết", "weather", "forecast"), ("thời tiết", "weather", "forecast", "temperature", "nhiệt độ")),
    (("đặt phòng", "phòng học nhóm", "study room"), ("đặt phòng", "phòng học nhóm", "study room", "room booking")),
    (("chỗ ở", "accommodation", "housing", "ký túc"), ("chỗ ở", "housing", "student accommodation", "ký túc xá", "dormitory")),
    (("myrmit",), ("myrmit",)),
)
_AMOUNT_QUERY_MARKERS = ("bao nhiêu", "how much", "mức phí", "chi phí", "cost", "price")
_MONEY_RE = re.compile(
    r"(?:\d[\d.,]*\s?(?:vnd|vnđ|đ|₫|usd|aud|eur))|(?:\d{2,}[.,]\d{2,})",
    re.IGNORECASE,
)


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
        context_parts.append(f"[{' | '.join(labels)}]\n{content}")
    return "\n\n---\n\n".join(context_parts)


def _query_tokens(query: str) -> set[str]:
    folded = query.casefold()
    tokens = {
        token
        for token in _TOKEN_RE.findall(folded)
        if token not in _STOPWORDS
    }
    for phrase, aliases in _PHRASE_ALIASES.items():
        if phrase in folded:
            tokens.update(aliases)
    return tokens


def _has_topic_evidence(query: str, chunks: list[dict]) -> bool:
    """Reject clearly unsupported intents before extractive/LLM generation."""

    if not chunks:
        return False
    evidence = "\n".join(str(chunk.get("content", "")) for chunk in chunks).casefold()
    folded_query = query.casefold()
    if "myrmit" in folded_query and (
        "đăng ký" in folded_query or "học phần" in folded_query
    ):
        return any(
            marker in evidence
            for marker in (
                "đăng ký môn học qua myrmit",
                "course registration through myrmit",
                "course enrolment through myrmit",
            )
        )
    for query_markers, evidence_markers in _UNSUPPORTED_TOPIC_RULES:
        if any(marker in folded_query for marker in query_markers):
            return any(marker in evidence for marker in evidence_markers)
    return True


def _is_amount_question(query: str) -> bool:
    return any(marker in query.casefold() for marker in _AMOUNT_QUERY_MARKERS)


def _has_amount_evidence(query: str, chunks: list[dict]) -> bool:
    """Require a monetary value before answering an explicit price question."""

    if not _is_amount_question(query):
        return True
    return any(_MONEY_RE.search(str(chunk.get("content", ""))) for chunk in chunks)


def _amount_clarification(chunks: list[dict]) -> str:
    """Give a grounded next step when a price question lacks an exact amount."""

    citation = _citation_label(chunks[0]) if chunks else "tuition-fees-rmit.md"
    return (
        "Tài liệu hiện có không nêu một mức học phí chung cho toàn bộ RMIT Vietnam. "
        "Học phí được công bố theo từng chương trình; mức phí theo năm và tổng chương trình "
        "được đăng trên website RMIT. Hãy cho biết chương trình, bậc học và năm nhập học để "
        f"tôi tra đúng mức phí. [{citation}]"
    )


def _extractive_answer(query: str, chunks: list[dict]) -> str:
    """Produce a citation-bearing answer without an external LLM."""

    if not chunks:
        return "Tôi không thể xác minh thông tin này từ các nguồn hiện có."

    query_folded = query.casefold()
    query_tokens = _query_tokens(query)
    amount_query = any(marker in query_folded for marker in _AMOUNT_QUERY_MARKERS)
    query_phrases = [
        phrase
        for phrase in _PHRASE_ALIASES
        if phrase in query_folded and len(phrase.split()) > 1
    ]
    # Also keep English multi-word intents such as "academic achievement".
    query_phrases.extend(
        phrase
        for phrase in ("academic achievement", "study room", "room booking")
        if phrase in query_folded
    )
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
            sentence_folded = sentence.casefold()
            phrase_bonus = max(
                [0.15 if phrase in sentence_folded else 0.0 for phrase in query_phrases]
                + [0.15 if query_folded.strip() in sentence_folded else 0.0]
            )
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
        if amount_query and not _MONEY_RE.search(sentence):
            continue
        # Never use the first arbitrary sentence as an answer. That behavior
        # made out-of-domain questions such as weather look answered.
        if score >= 0.25:
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

        try:
            max_tokens = max(1, int(os.getenv("LLM_MAX_TOKENS", str(LLM_MAX_TOKENS))))
        except ValueError:
            max_tokens = LLM_MAX_TOKENS

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
            max_tokens=max_tokens,
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

    if (
        not chunks
        or "không thể xác minh" in answer.casefold()
        or re.search(r"\[[^\]]+\]", answer)
    ):
        return answer
    labels = list(
        dict.fromkeys(f"[{_citation_label(chunk, i)}]" for i, chunk in enumerate(chunks))
    )
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

    if not _has_topic_evidence(query.strip(), chunks):
        return {
            "answer": "Tôi không thể xác minh thông tin này từ các nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    reordered = reorder_for_llm(chunks)
    if _is_amount_question(query.strip()) and not _has_amount_evidence(query.strip(), reordered):
        return {
            "answer": _amount_clarification(reordered),
            "sources": reordered[:1],
            "retrieval_source": "pageindex"
            if reordered and all(chunk.get("source") == "pageindex" for chunk in reordered)
            else "hybrid",
        }

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
