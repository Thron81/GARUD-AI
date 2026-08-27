"""Ingest documents into a FAISS vector index for RAG retrieval.

Reads .txt and .md files from a source directory, splits them into
overlapping chunks, embeds each chunk with a sentence-transformers model,
and saves a FAISS index + a metadata JSON (chunk text + source file) so
retriever.py can look them up later.

Usage:
    py rag/ingest.py --source rag/data --index rag/index
"""

from __future__ import annotations

import argparse
import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800  # characters
CHUNK_OVERLAP = 120


def read_documents(source_dir: str) -> list[tuple[str, str]]:
    """Return a list of (relative_path, full_text) for every .txt/.md file."""
    docs = []
    for root, _dirs, files in os.walk(source_dir):
        for name in files:
            if name.lower().endswith((".txt", ".md")):
                path = os.path.join(root, name)
                rel = os.path.relpath(path, source_dir)
                with open(path, "r", encoding="utf-8", errors="replace") as handle:
                    docs.append((rel, handle.read()))
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple fixed-size sliding-window chunker with overlap.

    Splits on whitespace boundaries where possible to avoid cutting words
    in half. Good enough for a project-scale corpus; swap for a
    recursive/semantic splitter if the docs get more structured.
    """
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to end on a whitespace boundary rather than mid-word.
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, end) if end == len(text) else end - overlap
        if start <= 0:
            start = end
    return chunks


def build_index(source_dir: str, index_dir: str, embed_model_name: str) -> None:
    os.makedirs(index_dir, exist_ok=True)
    docs = read_documents(source_dir)
    if not docs:
        raise SystemExit(f"No .txt/.md files found under '{source_dir}'.")

    print(f"Loaded {len(docs)} document(s) from {source_dir}")
    embedder = SentenceTransformer(embed_model_name)

    all_chunks: list[str] = []
    metadata: list[dict[str, str]] = []
    for source, text in docs:
        for chunk in chunk_text(text):
            all_chunks.append(chunk)
            metadata.append({"source": source, "text": chunk})

    if not all_chunks:
        raise SystemExit("No non-empty chunks were produced from the source documents.")

    print(f"Embedding {len(all_chunks)} chunk(s)...")
    embeddings = embedder.encode(all_chunks, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")

    # Inner product on normalized vectors == cosine similarity.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump({"embed_model": embed_model_name, "chunks": metadata}, handle, indent=2)

    print(f"Saved index + metadata to {index_dir} ({len(all_chunks)} chunks, dim={embeddings.shape[1]})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS index from a directory of text/markdown files.")
    parser.add_argument("--source", default="rag/data", help="Directory of .txt/.md source documents.")
    parser.add_argument("--index", default="rag/index", help="Directory to write the FAISS index + metadata into.")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_index(args.source, args.index, args.embed_model)


if __name__ == "__main__":
    main()
