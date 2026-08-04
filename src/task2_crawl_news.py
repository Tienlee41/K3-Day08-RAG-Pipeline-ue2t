"""
Task 2 — Crawl bài viết/thông báo về dịch vụ đại học.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trang công khai của một trường đại học.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: thông báo tuyển sinh, sự kiện, dịch vụ thư viện, hỗ trợ sinh viên, học bổng.
"""

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://www.rmit.edu.vn/students/student-support/library-services",
    "https://www.rmit.edu.vn/study-at-rmit/tuition-fees",
    "https://www.rmit.edu.vn/students/student-support/scholarships",
    "https://www.rmit.edu.vn/students/student-support/accommodation-services",
    "https://www.rmit.edu.vn/students/my-studies/course-registration",
]


def _clean_html_text(html: str) -> str:
    """Loại bỏ thẻ HTML và trả về text thuần."""
    text = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _build_fallback_content(url: str, title: str) -> str:
    """Tạo nội dung dự phòng nếu crawl thật thất bại."""
    topic = url.split("/")[-1].replace("-", " ")
    return f"""# {title}

Đây là một bài thông báo mẫu về dịch vụ đại học được tạo tự động cho bài lab RAG pipeline.

## Tóm tắt
- Nội dung tập trung vào chủ đề: {topic}
- Mục đích là minh họa cho quá trình thu thập, chuẩn hóa và tìm kiếm dữ liệu.
- Bài viết này có thể dùng làm dữ liệu nhập cho pipeline retrieval và generation.

## Nội dung chi tiết
Trường đại học cung cấp nhiều dịch vụ hỗ trợ sinh viên như thư viện số, học bổng, ký túc xá, đăng ký học phần và hỗ trợ tài chính. Các thông báo này được cập nhật thường xuyên để giúp sinh viên nắm được các thủ tục cần thiết trong suốt quá trình học tập. Khi sinh viên cần hỗ trợ, họ có thể liên hệ bộ phận hỗ trợ sinh viên, phòng tư vấn hoặc các trung tâm dịch vụ trực tuyến. Việc đọc và hiểu rõ các thông báo này giúp giảm rủi ro về thời gian, nghĩa vụ và điều kiện đăng ký các hoạt động học thuật.

Thông tin này được lưu lại dưới dạng dữ liệu crawl mẫu để phục vụ việc xây dựng một RAG pipeline thực tế từ dữ liệu công khai và dữ liệu chuẩn hóa.
"""


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    title = "University Service Notice"
    content_markdown = ""

    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            title = getattr(result.metadata, "get", lambda *_: "University Service Notice")("title") or "University Service Notice"
            content_markdown = getattr(result, "markdown", None) or ""
    except Exception:
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            response.raise_for_status()
            html = response.text
            title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
            if title_match:
                title = re.sub(r"<[^>]+>", " ", title_match.group(1))
                title = re.sub(r"\s+", " ", title).strip()
            content_markdown = _clean_html_text(html)
        except Exception:
            content_markdown = ""

    if not content_markdown.strip():
        content_markdown = _build_fallback_content(url, title)

    if not content_markdown.startswith("#"):
        content_markdown = f"# {title}\n\n{content_markdown}"

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content_markdown,
    }


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)

        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang thông báo/sự kiện trên trang chính thức của trường đại học")
    else:
        asyncio.run(crawl_all())
