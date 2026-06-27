from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark TurboInfer /v1/completions over HTTP.")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--model", default=None)
    parser.add_argument("--engine", choices=["naive", "kv-cache", "continuous"], default="kv-cache")
    parser.add_argument("--prompt", default="Explain why KV cache improves LLM decoding.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--quiet", action="store_true")
    return parser


def post_completion(
    client: httpx.Client,
    url: str,
    model: str | None,
    engine: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, object]:
    started = time.perf_counter()
    payload: dict[str, object] = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "engine": engine,
        "temperature": 0.0,
    }
    if model is not None:
        payload["model"] = model
    response = client.post(url, json=payload)
    response.raise_for_status()
    elapsed = time.perf_counter() - started
    body = response.json()
    return {
        "elapsed_seconds": elapsed,
        "usage": body["usage"],
        "metrics": body["metrics"],
    }


def percentile(values: list[float], pct: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, round((pct / 100.0) * (len(sorted_values) - 1)))
    return sorted_values[idx]


def main() -> None:
    args = build_parser().parse_args()
    started = time.perf_counter()
    results: list[dict[str, object]] = []
    with httpx.Client(timeout=args.timeout_seconds) as client:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(
                    post_completion,
                    client,
                    args.url,
                    args.model,
                    args.engine,
                    f"{args.prompt} Request {idx}.",
                    args.max_tokens,
                )
                for idx in range(args.num_requests)
            ]
            completed = 0
            for future in as_completed(futures):
                results.append(future.result())
                completed += 1
                if not args.quiet:
                    print(f"completed {completed}/{args.num_requests}", flush=True)

    total_seconds = time.perf_counter() - started
    latencies = [float(result["elapsed_seconds"]) for result in results]
    completion_tokens = [
        int(result["usage"]["completion_tokens"])  # type: ignore[index]
        for result in results
    ]
    total_completion_tokens = sum(completion_tokens)
    summary = {
        "url": args.url,
        "engine": args.engine,
        "num_requests": args.num_requests,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "total_seconds": total_seconds,
        "request_throughput_per_second": args.num_requests / total_seconds,
        "completion_token_throughput_per_second": total_completion_tokens / total_seconds,
        "mean_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "p50_latency_seconds": percentile(latencies, 50),
        "p95_latency_seconds": percentile(latencies, 95),
        "total_completion_tokens": total_completion_tokens,
        "responses": results,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
