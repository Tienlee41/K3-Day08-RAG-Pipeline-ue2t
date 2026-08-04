"""Task 2 - Crawl bai viet/thong bao ve dich vu dai hoc.

The crawler stores one JSON document per source page in
``data/landing/news/``.  Crawl4AI is used when it is installed.  A small
``requests`` + standard-library HTML parser fallback is kept so that the
collection script is still usable in environments where the Chromium
browser binary has not been installed yet.

Install the preferred crawler with::

    pip install crawl4ai
    playwright install chromium

The fallback uses the ``requests`` dependency already declared in
``requirements.txt`` and never invents article content: if a page cannot be
downloaded or parsed, crawling that page fails instead of writing fake data.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

# Five public RMIT Vietnam pages covering student support, library services,
# scholarships and student activities.  Keep this list explicit so that a
# rerun produces a reproducible landing dataset.
ARTICLE_URLS = [
    "https://www.rmit.edu.vn/students/support/student-academic-success",
    "https://www.rmit.edu.vn/libraryvn/student-support",
    "https://www.rmit.edu.vn/news/all-news/2025/oct/rmit-vietnam-awards-47-5-billion-vnd-in-2025-scholarships",
    "https://www.rmit.edu.vn/students/student-news-and-events/student-news/2025/peer-assisted-learning-10-year-anniversary",
    "https://www.rmit.edu.vn/news/all-news/2025/dec/multiculturalism-and-student-creativity-shine-at-rmit-light-night",
]

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.8,vi;q=0.7",
    "Referer": "https://www.rmit.edu.vn/",
}


def setup_directory() -> None:
    """Create ``data/landing/news/`` if it does not exist."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _metadata_value(metadata: Any, key: str) -> str | None:
    """Read a metadata field from either a dict or a Crawl4AI object."""

    if isinstance(metadata, dict):
        value = metadata.get(key)
    else:
        value = getattr(metadata, key, None)
    return str(value).strip() if value else None


class _HTMLArticleParser(HTMLParser):
    """Extract readable text and basic metadata without another dependency.

    RMIT pages expose their main content in ``main``/``article`` elements.
    When a page does not, the parser falls back to visible body text.  Header,
    navigation, footer and script content are ignored to keep the landing
    document useful for Task 3 and subsequent retrieval tasks.
    """

    _SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form"}
    _BLOCK_TAGS = {
        "article",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "main",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "tr",
        "ul",
    }
    _FOCUS_CLASS_MARKERS = {
        "article-body",
        "article-content",
        "article-header",
        "content-header-content",
        "intro",
        "text-component-inner",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.published_date = ""
        self._title_depth = 0
        self._skip_depth = 0
        self._main_depth = 0
        self._has_main = False
        self._focus_depth = 0
        self._open_tags: list[tuple[str, bool]] = []
        self._has_focus = False
        self._body_parts: list[str] = []
        self._main_parts: list[str] = []
        self._focus_parts: list[str] = []
        self._title_parts: list[str] = []

    @staticmethod
    def _attrs(attrs: list[tuple[str, str | None]]) -> dict[str, str]:
        return {key.lower(): value or "" for key, value in attrs}

    def _active_parts(self) -> list[str]:
        if self._focus_depth:
            return self._focus_parts
        return self._main_parts if self._main_depth else self._body_parts

    def _close_open_tag(self, tag: str) -> None:
        """Close the latest matching tag and update focus nesting."""

        for index in range(len(self._open_tags) - 1, -1, -1):
            open_tag, was_focus_container = self._open_tags[index]
            if open_tag != tag:
                continue
            self._open_tags.pop(index)
            if was_focus_container:
                self._focus_depth = max(0, self._focus_depth - 1)
            break

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = self._attrs(attrs)

        if tag == "title":
            self._title_depth = 1
        elif tag == "meta":
            key = (
                attributes.get("property")
                or attributes.get("name")
                or attributes.get("itemprop")
                or ""
            ).lower()
            content = attributes.get("content", "").strip()
            if content and key in {
                "og:title",
                "twitter:title",
            } and not self.title:
                self.title = content
            if content and key in {
                "article:published_time",
                "date",
                "datepublished",
                "publishdate",
            } and not self.published_date:
                self.published_date = content

        classes = set(attributes.get("class", "").lower().split())
        is_focus_container = any(
            marker in class_name
            for class_name in classes
            for marker in self._FOCUS_CLASS_MARKERS
        )
        self._open_tags.append((tag, is_focus_container))

        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return

        if is_focus_container:
            self._focus_depth += 1
            self._has_focus = True

        if tag in {"main", "article"}:
            self._main_depth += 1
            self._has_main = True

        if self._skip_depth == 0 and tag in self._BLOCK_TAGS:
            parts = self._active_parts()
            parts.append("\n")
            if tag.startswith("h") and len(tag) == 2:
                parts.append("#" * int(tag[1]) + " ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._title_depth = 0
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            self._close_open_tag(tag)
            return
        if self._skip_depth == 0 and tag in self._BLOCK_TAGS:
            self._active_parts().append("\n")
        if tag in {"main", "article"} and self._main_depth:
            self._main_depth -= 1
        self._close_open_tag(tag)

    def handle_data(self, data: str) -> None:
        if self._title_depth:
            self._title_parts.append(data)
        if self._skip_depth == 0 and data.strip():
            self._active_parts().append(data)

    def finish(self) -> tuple[str, str, str]:
        if not self.title:
            self.title = " ".join(self._title_parts)
        focus_content = _normalise_text("".join(self._focus_parts))
        main_parts = self._main_parts if self._has_main and self._main_parts else self._body_parts
        main_content = _normalise_text("".join(main_parts))
        # Article pages have a generic ``main`` wrapper containing related-news
        # cards as well as the article.  Prefer the focused article/text
        # components when available; otherwise retain the main/body fallback.
        content = focus_content if self._has_focus and len(focus_content) >= 200 else main_content
        return self.title.strip(), self.published_date.strip(), content


def _normalise_text(value: str) -> str:
    """Collapse HTML whitespace while retaining useful paragraph breaks."""

    lines: list[str] = []
    for raw_line in value.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if line and (not lines or lines[-1] != line):
            lines.append(line)

    return "\n\n".join(lines).strip()


def _slug_for_url(url: str, index: int) -> str:
    """Return a readable, filesystem-safe and deterministic JSON basename."""

    slug = urlparse(url).path.rstrip("/").split("/")[-1]
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", slug).strip("-").lower()
    return f"article_{index:02d}_{slug or 'page'}.json"


def _crawl_with_requests(url: str) -> dict[str, str]:
    """Download and parse one page using requests and the stdlib parser."""

    import requests

    # Some CDN edges temporarily return 405/429 for a perfectly valid page.
    # A trailing-slash retry is safe for these RMIT pages and keeps a transient
    # edge response from leaving a half-populated landing directory.
    candidate_urls = [url, url.rstrip("/") + "/"]
    response = None
    last_error: Exception | None = None
    for attempt, candidate_url in enumerate(candidate_urls):
        try:
            response = requests.get(candidate_url, headers=REQUEST_HEADERS, timeout=30)
            if response.status_code in {403, 405, 429, 500, 502, 503, 504} and attempt < len(candidate_urls) - 1:
                time.sleep(1)
                continue
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt < len(candidate_urls) - 1:
                time.sleep(1)
                continue
            raise
    if response is None:
        raise RuntimeError(f"No response received from {url}: {last_error}")
    parser = _HTMLArticleParser()
    parser.feed(response.text)
    parser.close()
    title, published_date, content = parser.finish()
    if not title:
        title = urlparse(url).path.rstrip("/").split("/")[-1].replace("-", " ").title()
    if not content:
        raise RuntimeError("The page did not contain readable article content")
    return {
        "title": title,
        "date_published": published_date,
        "content_markdown": content,
    }


async def crawl_article(url: str) -> dict[str, str]:
    """Crawl one article and return source metadata plus Markdown content.

    Crawl4AI is attempted first.  If it is unavailable or its browser cannot
    start, the HTTP parser fallback is used.  Both paths return the same JSON
    schema so downstream tasks do not depend on the installed crawler.
    """

    crawled: dict[str, str] | None = None
    crawl4ai_error: Exception | None = None

    try:
        from crawl4ai import AsyncWebCrawler

        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
        metadata = getattr(result, "metadata", {}) or {}
        title = _metadata_value(metadata, "title") or ""
        content = str(getattr(result, "markdown", "") or "").strip()
        if content:
            crawled = {
                "title": title,
                "date_published": "",
                "content_markdown": content,
            }
    except Exception as exc:  # Browser setup can fail before arun is called.
        crawl4ai_error = exc

    if crawled is None:
        try:
            crawled = await asyncio.to_thread(_crawl_with_requests, url)
        except Exception as exc:
            detail = f"; Crawl4AI error: {crawl4ai_error}" if crawl4ai_error else ""
            raise RuntimeError(f"Could not crawl {url}: {exc}{detail}") from exc

    title = crawled["title"].strip() or "University service article"
    content = crawled["content_markdown"].strip()
    if not content.startswith(f"# {title}"):
        content = f"# {title}\n\n{content}"

    article = {
        "url": url,
        "title": title,
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": content,
    }
    if crawled.get("date_published"):
        article["date_published"] = crawled["date_published"]
    return article


async def crawl_all() -> list[Path]:
    """Crawl all configured URLs and save one UTF-8 JSON file per article."""

    setup_directory()
    if len(ARTICLE_URLS) < 5:
        raise ValueError("Task 2 requires at least five article URLs")

    saved_files: list[Path] = []
    for index, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{index}/{len(ARTICLE_URLS)}] Crawling: {url}")
        article = await crawl_article(url)
        filepath = DATA_DIR / _slug_for_url(url, index)
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        saved_files.append(filepath)
        print(f"  Saved: {filepath}")
    return saved_files


if __name__ == "__main__":
    asyncio.run(crawl_all())
