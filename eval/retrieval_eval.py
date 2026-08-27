"""Evaluate retrieval quality: precision/recall@k against a labeled query set.

Labeled set format (JSON list):
    [
      {"query": "how do I reset the conversation?", "expected_source": "README.md"},
      ...
    ]

A retrieval counts as a hit if expected_source appears anywhere in the
top-k results for that query.

Usage:
    py eval/retrieval_eval.py --labels eval/retrieval_labels.json --k 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.retriever import Retriever  # noqa: E402


def evaluate(labels_path: str, index_dir: str, k: int) -> dict:
    with open(labels_path, "r", encoding="utf-8") as handle:
        labels = json.load(handle)

    retriever = Retriever(index_dir)
    hits = 0
    per_query = []

    for item in labels:
        query = item["query"]
        expected = item["expected_source"]
        results = retriever.retrieve(query, top_k=k)
        found_sources = [r.source for r in results]
        hit = expected in found_sources
        hits += int(hit)
        per_query.append(
            {
                "query": query,
                "expected_source": expected,
                "retrieved_sources": found_sources,
                "hit": hit,
            }
        )

    total = len(labels)
    precision_at_k = hits / total if total else 0.0  # here: fraction of queries with a hit in top-k
    return {
        "k": k,
        "total_queries": total,
        "hits": hits,
        "hit_rate_at_k": precision_at_k,
        "per_query": per_query,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval against a labeled query set.")
    parser.add_argument("--labels", default="eval/retrieval_labels.json")
    parser.add_argument("--index", default="rag/index")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", default="eval/results/retrieval_eval.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = evaluate(args.labels, args.index, args.k)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"hit_rate@{args.k} = {results['hit_rate_at_k']:.2%} ({results['hits']}/{results['total_queries']})")
    print(f"Full results saved to {args.output}")


if __name__ == "__main__":
    main()
