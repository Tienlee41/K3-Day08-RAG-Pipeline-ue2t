"""Task 3 - Convert every landing document to Markdown.

The landing layer contains source PDFs and crawled JSON files.  This module
creates a text-only, UTF-8 Markdown layer in ``data/standardized/`` while
preserving the ``legal/`` and ``news/`` subdirectories.
"""

from __future__ import annotations

import json
from pathlib import Path

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
LEGAL_EXTENSIONS = {".pdf", ".docx", ".doc"}


def _get_markitdown():
    """Load MarkItDown lazily so news-only utilities remain importable."""

    try:
        from markitdown import MarkItDown
    except ImportError as exc:
        raise RuntimeError(
            'MarkItDown or one of its conversion dependencies could not be '
            'imported. Install with: pip install "markitdown[pdf]". '
            f"Details: {exc}"
        ) from exc
    return MarkItDown()


def _write_markdown(path: Path, content: str) -> Path:
    """Write normalized UTF-8 Markdown and return the output path."""

    content = content.strip()
    if not content:
        raise ValueError(f"Converted content is empty: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")
    return path


def convert_legal_docs() -> list[Path]:
    """Convert PDF/DOCX/DOC files under ``data/landing/legal/``.

    MarkItDown is used here because it provides one conversion API for PDF and
    Word documents and keeps the conversion step independent from indexing.
    """

    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    if not legal_dir.exists():
        return []

    converter = _get_markitdown()
    outputs: list[Path] = []
    for filepath in sorted(legal_dir.iterdir()):
        if not filepath.is_file() or filepath.suffix.lower() not in LEGAL_EXTENSIONS:
            continue
        print(f"Converting legal: {filepath.name}")
        result = converter.convert(str(filepath))
        output_path = output_dir / f"{filepath.stem}.md"
        outputs.append(_write_markdown(output_path, result.text_content))
        print(f"  Saved: {output_path}")
    return outputs


def convert_news_articles() -> list[Path]:
    """Convert crawled JSON articles and retain source metadata in Markdown."""

    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    if not news_dir.exists():
        return []

    outputs: list[Path] = []
    for filepath in sorted(news_dir.iterdir()):
        if not filepath.is_file() or filepath.suffix.lower() != ".json":
            continue
        print(f"Converting news: {filepath.name}")
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid crawled JSON: {filepath}") from exc

        title = str(data.get("title") or "Untitled university news article").strip()
        content = str(data.get("content_markdown") or "").strip()
        if not content:
            raise ValueError(f"Missing content_markdown in {filepath}")

        # Metadata is deliberately part of the standardized text: later
        # retrieval results can cite the original URL without reopening JSON.
        header = (
            f"# {title}\n\n"
            f"**Source:** {data.get('url', 'N/A')}\n"
            f"**Crawled:** {data.get('date_crawled', 'N/A')}\n"
        )
        if data.get("date_published"):
            header += f"**Published:** {data['date_published']}\n"
        header += "\n---\n\n"

        output_path = output_dir / f"{filepath.stem}.md"
        outputs.append(_write_markdown(output_path, header + content))
        print(f"  Saved: {output_path}")
    return outputs


def convert_all() -> dict[str, list[Path]]:
    """Convert legal and news landing data, preserving their subdirectories."""

    print("=" * 50)
    print("Task 3: Convert to Markdown")
    print("=" * 50)
    legal_outputs = convert_legal_docs()
    news_outputs = convert_news_articles()
    print(f"Done: {len(legal_outputs)} legal + {len(news_outputs)} news Markdown files")
    return {"legal": legal_outputs, "news": news_outputs}


if __name__ == "__main__":
    convert_all()
