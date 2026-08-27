"""Query a FAISS index built by ingest.py and return top-k chunks with sources."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedChunk:
    text: str
    source: str
    score: float


class Retriever:
    """Loads an index once; call .retrieve(query) as many times as needed."""

    def __init__(self, index_dir: str = "rag/index") -> None:
        index_path = os.path.join(index_dir, "index.faiss")
        metadata_path = os.path.join(index_dir, "metadata.json")
        if not os.path.isfile(index_path) or not os.path.isfile(metadata_path):
            raise FileNotFoundError(
                f"No index found at '{index_dir}'. Run rag/ingest.py first to build one."
            )

        self.index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.chunks: list[dict[str, str]] = payload["chunks"]
        self.embedder = SentenceTransformer(payload["embed_model"])

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievedChunk]:
        query_vec = self.embedder.encode([query], normalize_embeddings=True)
        query_vec = np.asarray(query_vec, dtype="float32")
        scores, indices = self.index.search(query_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            entry = self.chunks[idx]
            results.append(RetrievedChunk(text=entry["text"], source=entry["source"], score=float(score)))
        return results

    @staticmethod
    def format_context(chunks: list[RetrievedChunk]) -> str:
        """Format retrieved chunks into a context block with citation tags."""
        parts = []
        for chunk in chunks:
            parts.append(f"[source: {chunk.source}]\n{chunk.text}")
        return "\n\n".join(parts)
