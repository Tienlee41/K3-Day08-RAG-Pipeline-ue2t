"""University Services RAG Chatbot.

Run from the repository root:

    streamlit run app.py
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="University Services RAG Chatbot",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

SUGGESTIONS = [
    "RMIT có các phương thức thanh toán học phí nào?",
    "Nếu trả sách thư viện trễ hạn thì sao?",
    "Nếu sinh viên nhận học bổng trượt một môn học, ai sẽ trả học phí môn đó?",
    "Học bổng có thể chuyển đổi thành tiền mặt hoặc chuyển nhượng không?",
    "Học bổng cho sinh viên hiện tại có thể bảo lưu trong bao lâu?",
    "Chính sách Family Tuition Fee Assistance có giảm 5% học phí không?",
]


def _contextualise_follow_up(query: str, history: list[dict]) -> str:
    """Resolve short/pronominal follow-ups before sending them to Task 10."""

    user_turns = [item.get("content", "") for item in history if item.get("role") == "user"]
    if not user_turns:
        return query
    follow_up = re.search(
        r"\b(it|that|this|they|them|those|these|what about|how about)\b"
        r"|\b(còn|này|đó|đấy|vậy|thế|thêm)\b",
        query,
        re.IGNORECASE,
    )
    if follow_up or len(query.split()) <= 7:
        previous_answer = next(
            (item.get("content", "") for item in reversed(history) if item.get("role") == "assistant"),
            "",
        )
        return (
            f"Previous user question: {user_turns[-1]}\n"
            f"Previous answer: {previous_answer[-1200:]}\n"
            f"Current follow-up question: {query}"
        )
    return query


def _render_sources(sources: list[dict], key_prefix: str) -> None:
    if not sources:
        return
    with st.expander(f"📚 Nguồn tham khảo ({len(sources)} đoạn)", expanded=False):
        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata", {}) or {}
            source_name = metadata.get("source") or metadata.get("filename") or "Unknown"
            doc_type = metadata.get("type", "unknown")
            section = metadata.get("section")
            try:
                score = float(source.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                score = 0.0
            title = f"[{index}] {source_name} · {doc_type} · score {score:.4f}"
            if section:
                title += f" · {section}"
            st.markdown(f"**{title}**")
            st.caption(f"Citation: `[{source_name}]`")
            content = str(source.get("content", "")).strip()
            st.text_area(
                "Evidence",
                content[:1200] + ("…" if len(content) > 1200 else ""),
                height=120,
                key=f"{key_prefix}_{index}",
                label_visibility="collapsed",
            )
            if index < len(sources):
                st.divider()


def _render_message(message: dict, index: int) -> None:
    with st.chat_message(message.get("role", "assistant")):
        st.markdown(str(message.get("content", "")))
        if message.get("role") == "assistant":
            _render_sources(message.get("sources", []), f"history_{index}")


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

with st.sidebar:
    st.title("🎓 University Services RAG")
    st.caption("Hỏi đáp về học phí, học bổng và dịch vụ đại học")
    st.divider()

    st.subheader("💡 Câu hỏi gợi ý")
    for index, suggestion in enumerate(SUGGESTIONS):
        if st.button(suggestion, use_container_width=True, key=f"suggestion_{index}"):
            st.session_state.pending_query = suggestion

    st.divider()
    st.subheader("⚙️ Thiết lập")
    top_k = st.slider("Số chunks đưa vào context", min_value=3, max_value=10, value=5)
    if st.button("🧹 Xóa cuộc trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    mode = "OpenRouter/OpenAI" if os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") else "Offline extractive"
    st.caption(f"Generation mode: **{mode}**")
    st.caption("Task 9 Retrieval → Task 10 Generation → Citation")
    if st.session_state.messages:
        st.caption(f"Conversation memory: {len(st.session_state.messages)} lượt")

st.title("🎓 University Services RAG Chatbot")
st.caption("Trả lời dựa trên tài liệu chính sách/dịch vụ đại học và hiển thị nguồn kiểm chứng.")

for index, message in enumerate(st.session_state.messages):
    _render_message(message, index)

user_input = st.chat_input("Nhập câu hỏi của bạn…")
query = user_input or st.session_state.pending_query
if query:
    st.session_state.pending_query = None
    query = str(query).strip()
    history = [
        {"role": item.get("role"), "content": item.get("content", "")}
        for item in st.session_state.messages
        if item.get("role") in {"user", "assistant"}
    ]
    retrieval_query = _contextualise_follow_up(query, history)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất tài liệu và tổng hợp câu trả lời…"):
            try:
                from src.task10_generation import generate_with_citation

                response = generate_with_citation(retrieval_query, top_k=top_k)
                answer = response.get("answer", "Tôi chưa thể trả lời câu hỏi này.")
                sources = response.get("sources", [])
            except Exception as exc:
                answer = f"⚠️ Không thể chạy RAG pipeline: `{exc}`"
                sources = []
            st.markdown(answer)
            _render_sources(sources, f"current_{len(st.session_state.messages)}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
