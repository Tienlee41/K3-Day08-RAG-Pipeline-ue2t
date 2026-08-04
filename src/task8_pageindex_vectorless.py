"""Task 8 - PageIndex vectorless retrieval.

When PageIndex Cloud is configured, this module uploads the available PDF
documents and queries the legacy structured-retrieval endpoint.  The endpoint
is still useful here because Task 8 needs a list of evidence nodes rather than
only a generated answer.

For local development and automated tests, a deterministic structural fallback
is provided.  It reads whole Markdown documents/sections and scores heading
and term coverage without embeddings or chunking.  Results keep the same
``source='pageindex'`` marker, so the rest of the retrieval pipeline does not
need to know whether cloud or local vectorless retrieval was used.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = ROOT_DIR / "data" / "standardized"
LANDING_DIR = ROOT_DIR / "data" / "landing"
DOC_IDS_PATH = ROOT_DIR / "pageindex_doc_ids.json"

PAGEINDEX_API_URL = os.getenv("PAGEINDEX_API_URL", "https://api.pageindex.ai").rstrip("/")
PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")

_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")


def _api_key() -> str:
    """Read the key at call time so tests/apps may configure it dynamically."""

    return os.getenv("PAGEINDEX_API_KEY", PAGEINDEX_API_KEY).strip()


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(text).casefold()))


def _read_doc_id_cache() -> dict[str, str]:
    if not DOC_IDS_PATH.exists():
        return {}
    try:
        payload = json.loads(DOC_IDS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}

    if not isinstance(payload, dict):
        return {}
    documents = payload.get("documents", payload)
    if not isinstance(documents, dict):
        return {}
    result: dict[str, str] = {}
    for source, value in documents.items():
        if isinstance(value, dict):
            doc_id = value.get("doc_id") or value.get("id")
        else:
            doc_id = value
        if doc_id:
            result[str(source)] = str(doc_id)
    return result


def _write_doc_id_cache(doc_ids: dict[str, str]) -> None:
    DOC_IDS_PATH.write_text(
        json.dumps({"documents": doc_ids}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _pdf_files() -> list[Path]:
    """Return source PDFs supported by PageIndex Cloud."""

    files: set[Path] = set()
    if LANDING_DIR.exists():
        files.update(path for path in LANDING_DIR.rglob("*.pdf") if path.is_file())
    # Allow a user to place converted Markdown/PDF files in the ignored cache
    # directory without changing the tracked corpus.
    upload_dir = ROOT_DIR / "pageindex_pdfs"
    if upload_dir.exists():
        files.update(path for path in upload_dir.rglob("*.pdf") if path.is_file())
    return sorted(files)


def _pageindex_client():
    """Create the current SDK client, with compatibility for older releases."""

    if not _api_key():
        raise RuntimeError("PAGEINDEX_API_KEY is not configured")
    try:
        from pageindex import PageIndexClient
    except ImportError:
        try:
            from pageindex.client import PageIndexClient
        except ImportError as exc:
            raise RuntimeError(
                "Install the PageIndex SDK with: pip install -U pageindex"
            ) from exc
    return PageIndexClient(api_key=_api_key())


def upload_documents() -> dict[str, str]:
    """Upload available PDFs and persist ``source -> doc_id`` mappings.

    PageIndex Cloud currently accepts PDF uploads.  The repository already
    contains the three legal source PDFs; Markdown-only news documents remain
    searchable through the local structural fallback.
    """

    client = _pageindex_client()
    cached = _read_doc_id_cache()
    uploaded: dict[str, str] = dict(cached)
    pdf_files = _pdf_files()
    if not pdf_files:
        return uploaded

    for pdf_file in pdf_files:
        source = pdf_file.relative_to(ROOT_DIR).as_posix()
        if source in uploaded:
            continue
        response = client.submit_document(str(pdf_file))
        if not isinstance(response, dict):
            raise RuntimeError(f"Unexpected PageIndex upload response for {pdf_file.name}")
        doc_id = response.get("doc_id") or response.get("id")
        if not doc_id:
            raise RuntimeError(f"PageIndex did not return doc_id for {pdf_file.name}")
        uploaded[source] = str(doc_id)
        print(f"Uploaded {pdf_file.name} -> {doc_id}")

    _write_doc_id_cache(uploaded)
    return uploaded


def _configured_doc_ids() -> list[str]:
    explicit = os.getenv("PAGEINDEX_DOC_IDS", "") or os.getenv("PAGEINDEX_DOC_ID", "")
    if explicit:
        return [item.strip() for item in explicit.split(",") if item.strip()]
    return list(dict.fromkeys(_read_doc_id_cache().values()))


def _flatten_items(value: Any) -> Iterator[dict]:
    """Flatten both old nested and current flat relevant-content schemas."""

    if isinstance(value, dict):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _flatten_items(item)


def _parse_retrieval_payload(payload: dict, top_k: int, doc_id: str | None = None) -> list[dict]:
    results: list[dict] = []
    seen: set[str] = set()
    nodes = payload.get("retrieved_nodes", [])
    for node_index, node in enumerate(nodes if isinstance(nodes, list) else []):
        if not isinstance(node, dict):
            continue
        section = node.get("title") or node.get("section_title") or ""
        relevant_contents = node.get("relevant_contents", node.get("contents", []))
        for item in _flatten_items(relevant_contents):
            content = (
                item.get("relevant_content")
                or item.get("content")
                or item.get("text")
                or ""
            )
            if not isinstance(content, str) or not content.strip():
                continue
            content = content.strip()
            if content in seen:
                continue
            seen.add(content)
            rank = len(results)
            metadata = {
                "section": item.get("section_title") or section,
                "page_index": item.get("page_index", item.get("page")),
            }
            if doc_id:
                metadata["doc_id"] = doc_id
            metadata = {key: value for key, value in metadata.items() if value not in (None, "")}
            results.append(
                {
                    "content": content,
                    # PageIndex retrieval does not expose a comparable score;
                    # rank-based confidence is explicit and deterministic.
                    "score": 1.0 / (rank + 1),
                    "metadata": metadata,
                    "source": "pageindex",
                }
            )
            if len(results) >= top_k:
                return results
    return results


def _cloud_search(query: str, top_k: int, doc_ids: list[str]) -> list[dict]:
    """Query PageIndex's structured legacy retrieval endpoint."""

    import requests

    timeout = float(os.getenv("PAGEINDEX_TIMEOUT", "20"))
    headers = {"api_key": _api_key(), "Content-Type": "application/json"}
    results: list[dict] = []
    for doc_id in doc_ids:
        response = requests.post(
            f"{PAGEINDEX_API_URL}/retrieval/",
            headers=headers,
            json={"doc_id": doc_id, "query": query, "thinking": False},
            timeout=timeout,
        )
        response.raise_for_status()
        initial = response.json()
        retrieval_id = initial.get("retrieval_id") or initial.get("id")
        if not retrieval_id:
            results.extend(_parse_retrieval_payload(initial, top_k - len(results), doc_id))
            continue

        payload: dict = initial
        for _ in range(max(1, int(os.getenv("PAGEINDEX_POLL_ATTEMPTS", "10")))):
            if str(payload.get("status", "")).casefold() in {"completed", "complete", "succeeded"}:
                break
            if str(payload.get("status", "")).casefold() in {"failed", "error"}:
                break
            time.sleep(float(os.getenv("PAGEINDEX_POLL_INTERVAL", "1")))
            poll_response = requests.get(
                f"{PAGEINDEX_API_URL}/retrieval/{retrieval_id}/",
                headers={"api_key": _api_key()},
                timeout=timeout,
            )
            poll_response.raise_for_status()
            payload = poll_response.json()
        results.extend(_parse_retrieval_payload(payload, top_k - len(results), doc_id))
        if len(results) >= top_k:
            break
    return results[:top_k]


def _load_structural_sections() -> list[dict]:
    """Build whole-document/heading sections without chunking or embeddings."""

    sections: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return sections

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if not md_file.is_file():
            continue
        document = md_file.read_text(encoding="utf-8").strip()
        if not document:
            continue
        relative_path = md_file.relative_to(STANDARDIZED_DIR).as_posix()
        doc_type = relative_path.split("/", 1)[0] if "/" in relative_path else "unknown"
        lines = document.splitlines()
        current_title = md_file.stem.replace("-", " ")
        current_lines: list[str] = []
        found_heading = False

        def flush() -> None:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append(
                    {
                        "content": body,
                        "section": current_title,
                        "source": relative_path,
                        "filename": md_file.name,
                        "type": doc_type,
                    }
                )

        for line in lines:
            heading_match = _HEADING_RE.match(line)
            if heading_match:
                if current_lines:
                    flush()
                    current_lines = []
                found_heading = True
                current_title = heading_match.group(2).strip()
                current_lines.append(line.strip())
            else:
                current_lines.append(line)
        if current_lines:
            flush()
        if not found_heading and not current_lines and document:
            sections.append(
                {
                    "content": document,
                    "section": md_file.stem.replace("-", " "),
                    "source": relative_path,
                    "filename": md_file.name,
                    "type": doc_type,
                }
            )
    return sections


def _local_structural_search(query: str, top_k: int) -> list[dict]:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored: list[tuple[float, int, dict]] = []
    for order, section in enumerate(_load_structural_sections()):
        content_tokens = _tokenize(section["content"])
        title_tokens = _tokenize(section["section"])
        coverage = len(query_tokens & content_tokens) / len(query_tokens)
        title_coverage = len(query_tokens & title_tokens) / len(query_tokens)
        phrase_bonus = 0.15 if query.casefold().strip() in section["content"].casefold() else 0.0
        score = min(1.0, 0.65 * coverage + 0.25 * title_coverage + phrase_bonus)
        if score <= 0.0:
            continue
        scored.append((score, order, section))

    scored.sort(key=lambda item: (-item[0], item[1]))
    results: list[dict] = []
    for score, _, section in scored[:top_k]:
        results.append(
            {
                "content": section["content"],
                "score": float(score),
                "metadata": {
                    "source": section["source"],
                    "filename": section["filename"],
                    "type": section["type"],
                    "section": section["section"],
                    "mode": "local-structural",
                },
                "source": "pageindex",
            }
        )
    return results


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve vectorlessly via PageIndex Cloud or local structure fallback."""

    if not isinstance(query, str) or not query.strip() or not isinstance(top_k, int) or top_k <= 0:
        return []

    api_key = _api_key()
    if api_key:
        try:
            doc_ids = _configured_doc_ids()
            if not doc_ids:
                doc_ids = list(upload_documents().values())
            if doc_ids:
                cloud_results = _cloud_search(query.strip(), top_k, doc_ids)
                if cloud_results:
                    return cloud_results[:top_k]
        except Exception:
            # Cloud setup, processing, or network failures should not make the
            # fallback path unusable in a local lab.
            pass

    return _local_structural_search(query.strip(), top_k)


if __name__ == "__main__":
    if PAGEINDEX_API_KEY:
        print("PageIndex API mode enabled")
    else:
        print("PageIndex API key not set; using local structural mode")
    for result in pageindex_search("tuition fee payment methods", top_k=3):
        print(f"[{result['score']:.3f}] [{result['source']}] {result['content'][:100]}...")
