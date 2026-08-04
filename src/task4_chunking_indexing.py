"""Task 4 - Chunk standardized Markdown and index it in ChromaDB.

Technical choices:
* ``RecursiveCharacterTextSplitter`` with 800 characters and 120 overlap
  respects paragraph/sentence boundaries where possible while keeping chunks
  small enough for retrieval.  The overlap prevents a policy rule split at a
  boundary from losing its surrounding context.
* ``all-MiniLM-L6-v2`` produces 384-dimensional embeddings locally.  The
  current RMIT corpus is predominantly English; this model is lightweight and
  fast enough for a reproducible local lab.  A multilingual model such as
  ``BAAI/bge-m3`` can replace it later without changing the Chroma interface.
* ChromaDB is persistent, local and uses cosine distance, so Task 5 can query
  the same vectors without a server or external API key.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# 800 chars is large enough to keep a short policy procedure together while
# remaining focused for retrieval.  120 chars (~15%) carries context across a
# split without creating too many duplicate vectors.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
CHUNKING_METHOD = "recursive"

# The corpus currently contains primarily English RMIT material.  This model
# is 384-dimensional and inexpensive to run locally; the config can be changed
# to BAAI/bge-m3 (1024 dim) for a larger multilingual deployment.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

VECTOR_STORE = "chromadb"
COLLECTION_NAME = "university_services_docs"

_EMBEDDING_MODEL_INSTANCE: Any | None = None


def load_documents() -> list[dict]:
    """Load all standardized Markdown files with stable source metadata."""

    if not STANDARDIZED_DIR.exists():
        return []

    documents: list[dict] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if not md_file.is_file():
            continue
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if relative_path.parts else "unknown"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": relative_path.as_posix(),
                    "filename": md_file.name,
                    "type": doc_type,
                },
            }
        )
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents recursively and attach a deterministic chunk index."""

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        RecursiveCharacterTextSplitter = None

    def fallback_split(content: str) -> list[tuple[str, int]]:
        """Dependency-free character splitter used only when LangChain is absent."""

        if not content:
            return []
        step = max(1, CHUNK_SIZE - CHUNK_OVERLAP)
        chunks_with_offsets: list[tuple[str, int]] = []
        start = 0
        while start < len(content):
            end = min(len(content), start + CHUNK_SIZE)
            chunk = content[start:end].strip()
            if chunk:
                actual_start = content.find(chunk, start, end)
                chunks_with_offsets.append((chunk, actual_start if actual_start >= 0 else start))
            if end >= len(content):
                break
            start += step
        return chunks_with_offsets

    splitter = None
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            # Prefer Markdown sections, paragraphs and sentences before splitting
            # individual words or characters.
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""],
            length_function=len,
            add_start_index=True,
        )

    chunks: list[dict] = []
    for document in documents:
        content = str(document.get("content", "")).strip()
        if not content:
            continue
        metadata = dict(document.get("metadata", {}))
        if splitter is not None:
            splits = splitter.create_documents([content], metadatas=[metadata])
            split_items = [
                (split.page_content.strip(), int(split.metadata.get("start_index", 0)), split.metadata)
                for split in splits
            ]
        else:
            split_items = [
                (chunk_text, start_index, metadata)
                for chunk_text, start_index in fallback_split(content)
            ]

        for chunk_index, (chunk_text, start_index, split_metadata) in enumerate(split_items):
            if not chunk_text:
                continue
            chunk_metadata = dict(split_metadata)
            chunk_metadata["chunk_index"] = chunk_index
            chunk_metadata["chunk_start"] = int(start_index)
            chunk_metadata.pop("start_index", None)
            chunks.append({"content": chunk_text, "metadata": chunk_metadata})
    return chunks


def get_embedding_model():
    """Load and cache the sentence-transformers model used by Tasks 4 and 5."""

    global _EMBEDDING_MODEL_INSTANCE
    if _EMBEDDING_MODEL_INSTANCE is not None:
        return _EMBEDDING_MODEL_INSTANCE
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Task 4 requires sentence-transformers; install it with "
            "pip install sentence-transformers"
        ) from exc
    _EMBEDDING_MODEL_INSTANCE = SentenceTransformer(EMBEDDING_MODEL)
    return _EMBEDDING_MODEL_INSTANCE


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed chunks with normalized vectors and verify the configured dimension."""

    if not chunks:
        return []
    model = get_embedding_model()
    texts = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if len(embeddings.shape) != 2 or embeddings.shape[1] != EMBEDDING_DIM:
        raise RuntimeError(
            f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, "
            f"got {getattr(embeddings, 'shape', None)}"
        )

    embedded_chunks: list[dict] = []
    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunks.append(
            {
                **chunk,
                "metadata": dict(chunk.get("metadata", {})),
                "embedding": embedding.astype(float).tolist(),
            }
        )
    return embedded_chunks


def get_collection():
    """Open the persistent cosine-similarity collection used by retrieval."""

    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "Task 4 requires chromadb; install it with pip install chromadb"
        ) from exc

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _chunk_id(chunk: dict) -> str:
    """Create a stable ID that is safe even when a source filename changes."""

    metadata = chunk.get("metadata", {})
    source = str(metadata.get("source", "unknown"))
    index = str(metadata.get("chunk_index", 0))
    digest = hashlib.sha1(f"{source}:{index}".encode("utf-8")).hexdigest()[:16]
    return f"{digest}_{index}"


def index_to_vectorstore(chunks: list[dict]):
    """Replace the local collection with the current complete corpus."""

    if not chunks:
        raise ValueError("Cannot index an empty chunk list")
    if any("embedding" not in chunk for chunk in chunks):
        raise ValueError("Every chunk must be embedded before indexing")

    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError("Install chromadb before indexing") from exc

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # Replacing the generated collection prevents stale chunks from previous
    # corpora being mixed with the current standardized files.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    collection.upsert(
        ids=[_chunk_id(chunk) for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk.get("metadata", {}) for chunk in chunks],
    )
    return collection


def run_pipeline() -> dict[str, int]:
    """Run load -> chunk -> embed -> index and return counts."""

    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    documents = load_documents()
    print(f"Loaded {len(documents)} documents")
    if not documents:
        raise RuntimeError("No Markdown documents found; run Task 3 first")

    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")
    embedded_chunks = embed_chunks(chunks)
    print(f"Embedded {len(embedded_chunks)} chunks")
    collection = index_to_vectorstore(embedded_chunks)
    print(f"Indexed {collection.count()} chunks in {CHROMA_DIR}")
    return {"documents": len(documents), "chunks": len(embedded_chunks)}


if __name__ == "__main__":
    run_pipeline()
