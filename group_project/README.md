# Bài Tập Nhóm — University Services RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1: Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về dịch vụ và chính sách đại học liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

Xem code mẫu (DeepEval/RAGAS/TruLens) chi tiết trong `README.md` gốc mục "Yêu cầu 2".

### Deliverable Evaluation

- [ ] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [ ] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [ ] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [ ] So sánh A/B ít nhất 2 configs

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

```mermaid
flowchart TB
    user[Người dùng] --> ui[Streamlit Chatbot]
    ui --> memory[Conversation memory<br/>Chuẩn hoá câu hỏi follow-up]

    subgraph ingestion[Chuẩn bị dữ liệu]
        t1[Task 1<br/>Thu thập tài liệu chính sách]
        t2[Task 2<br/>Crawl tin tức/dịch vụ]
        raw[(data/landing)]
        t3[Task 3<br/>Chuyển đổi sang Markdown]
        docs[(data/standardized)]
        t4[Task 4<br/>Chunking & Chroma indexing]

        t1 --> raw
        t2 --> raw
        raw --> t3 --> docs --> t4
    end

    subgraph retrieval[Task 9 — Hybrid Retrieval]
        t5[Task 5<br/>Semantic search]
        t6[Task 6<br/>BM25 lexical search]
        t7[Task 7<br/>RRF / reranking]
        t8[Task 8<br/>PageIndex fallback]

        t4 --> t5
        docs --> t6
        docs --> t8
        t5 --> t7
        t6 --> t7
        t7 --> fallback{Evidence đủ tốt?}
        fallback -- Không --> t8
    end

    memory --> t5
    memory --> t6
    fallback -- Có --> context[Context chunks + metadata]
    t8 --> context
    context --> t10[Task 10<br/>Generation có citation]
    t10 --> llm[OpenRouter / OpenAI]
    llm --> answer[Trả lời + citation + source documents]
    answer --> ui
```

Pipeline kết hợp semantic search và BM25, rerank kết quả bằng RRF/cross-encoder, sau đó dùng PageIndex khi evidence yếu. Chatbot hiển thị câu trả lời, citation và các đoạn tài liệu đã dùng; OpenRouter/OpenAI là lớp sinh câu trả lời, còn chế độ extractive cục bộ là fallback khi API không sẵn sàng.

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| Nguyễn Văn Trường | 2A202601974 | Task 1, 2, 3, 4, 5 — thu thập dữ liệu, crawl, chuẩn hoá Markdown, chunking/indexing và semantic search | Hoàn thành |
| Lê Anh Tiến | 2A202601145 | Task 6, 7, 8, 9, 10 — lexical search, reranking, PageIndex, RAG pipeline, generation có citation; xây dựng và tích hợp chatbot Streamlit | Hoàn thành |

---

## Hướng Dẫn Chạy

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy app
streamlit run app.py
# hoặc
chainlit run app.py
```

---

## Lưu ý

Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
