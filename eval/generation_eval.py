"""Evaluate generation quality: RAG faithfulness (heuristic) + latency benchmarking.

Faithfulness here uses a lightweight lexical-overlap heuristic (no external
judge API, so it works fully offline): what fraction of the answer's
content words also appear in the retrieved context. This is a proxy, not
a ground-truth faithfulness score — call it out as such in your report,
and swap in an LLM-as-judge prompt later if you want a stronger signal.

Usage:
    py eval/generation_eval.py --benchmark-only --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.engine import GarudEngine  # noqa: E402

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on", "for",
    "and", "or", "it", "this", "that", "with", "as", "be", "by", "at", "from",
}


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def faithfulness_score(answer: str, context: str) -> float:
    """Fraction of the answer's content words that also appear in the context."""
    answer_words = _content_words(answer)
    if not answer_words:
        return 0.0
    context_words = _content_words(context)
    overlap = answer_words & context_words
    return len(overlap) / len(answer_words)


def run_faithfulness_eval(cases_path: str, engine: GarudEngine, output_path: str) -> None:
    """cases: JSON list of {"context": ..., "answer": ...} — e.g. saved RAG transcript turns."""
    with open(cases_path, "r", encoding="utf-8") as handle:
        cases = json.load(handle)

    results = []
    for case in cases:
        score = faithfulness_score(case["answer"], case["context"])
        results.append({**case, "faithfulness_score": score})

    avg = sum(r["faithfulness_score"] for r in results) / len(results) if results else 0.0
    payload = {"average_faithfulness": avg, "cases": results}

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"average_faithfulness = {avg:.2%} over {len(results)} case(s)")
    print(f"Full results saved to {output_path}")


def run_latency_benchmark(model_name: str, load_in_4bit: bool, prompts: list[str], output_path: str) -> None:
    engine = GarudEngine(model_name=model_name, load_in_4bit=load_in_4bit, max_new_tokens=200, temperature=0.0)
    engine.load(progress_callback=print)

    records = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": GarudEngine.system_prompt_for("chat")},
            {"role": "user", "content": prompt},
        ]
        start = time.perf_counter()
        reply = engine.generate_once(messages)
        elapsed = time.perf_counter() - start

        token_count = len(engine.tokenizer(reply).input_ids)
        records.append(
            {
                "prompt": prompt,
                "elapsed_seconds": elapsed,
                "output_tokens": token_count,
                "tokens_per_second": token_count / elapsed if elapsed > 0 else None,
            }
        )
        print(f"[{elapsed:.2f}s, {token_count} tok, {token_count/elapsed:.1f} tok/s] {prompt[:50]}")

    avg_tps = sum(r["tokens_per_second"] for r in records if r["tokens_per_second"]) / len(records)
    payload = {
        "model": model_name,
        "load_in_4bit": load_in_4bit,
        "device": str(engine.device),
        "average_tokens_per_second": avg_tps,
        "runs": records,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nAverage: {avg_tps:.1f} tok/s on {engine.device}. Full results saved to {output_path}")


DEFAULT_BENCHMARK_PROMPTS = [
    "Explain what a hash map is in two sentences.",
    "Write a Python function that reverses a string.",
    "Summarize the benefits of unit testing.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generation faithfulness and/or benchmark latency.")
    parser.add_argument("--cases", default="eval/faithfulness_cases.json", help="Path to faithfulness case JSON (skip with --benchmark-only).")
    parser.add_argument("--benchmark-only", action="store_true", help="Skip faithfulness eval, only run the latency benchmark.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--output-dir", default="eval/results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.benchmark_only and os.path.isfile(args.cases):
        engine = GarudEngine(model_name=args.model, load_in_4bit=args.load_in_4bit)
        run_faithfulness_eval(args.cases, engine, os.path.join(args.output_dir, "faithfulness_eval.json"))

    run_latency_benchmark(
        args.model, args.load_in_4bit, DEFAULT_BENCHMARK_PROMPTS,
        os.path.join(args.output_dir, "latency_benchmark.json"),
    )


if __name__ == "__main__":
    main()
