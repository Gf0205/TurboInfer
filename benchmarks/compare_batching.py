from __future__ import annotations

import argparse
import gc
import json
import time

import torch
from transformers import AutoTokenizer

from turboinfer.batching import StaticBatchKVCacheEngine
from turboinfer.engine import KVCacheEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare sequential KV decode with batched KV decode.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-requests", type=int, default=4)
    parser.add_argument("--prompt-token-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup-new-tokens", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def make_prompt(model_name: str, target_tokens: int, trust_remote_code: bool, suffix: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    seed_text = (
        "LLM serving systems use scheduling, batching, KV cache, memory bandwidth, "
        "and latency metrics to balance throughput and tail latency. "
    )
    prompt = f"{suffix}. {seed_text}"
    while len(tokenizer(prompt, return_tensors="pt")["input_ids"][0]) < target_tokens:
        prompt += seed_text
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0][:target_tokens]
    return tokenizer.decode(input_ids, skip_special_tokens=True)


def make_prompts(args: argparse.Namespace) -> list[str]:
    return [
        make_prompt(
            model_name=args.model,
            target_tokens=args.prompt_token_length,
            trust_remote_code=args.trust_remote_code,
            suffix=f"Request {idx}",
        )
        for idx in range(args.num_requests)
    ]


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_sequential(args: argparse.Namespace, prompts: list[str]) -> dict[str, object]:
    engine = KVCacheEngine(
        model_name=args.model,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    if args.warmup_new_tokens > 0:
        engine.generate(prompts[0], max_new_tokens=args.warmup_new_tokens)
        cleanup()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    results = [engine.generate(prompt, max_new_tokens=args.max_new_tokens) for prompt in prompts]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_seconds = end_time - start_time
    total_output_tokens = sum(result.metrics.output_tokens for result in results)
    metrics = {
        "model": args.model,
        "device": args.device,
        "num_requests": len(prompts),
        "prompt_tokens_per_request": results[0].metrics.prompt_tokens,
        "output_tokens_per_request": args.max_new_tokens,
        "total_output_tokens": total_output_tokens,
        "total_seconds": total_seconds,
        "request_throughput_per_second": len(prompts) / total_seconds,
        "token_throughput_per_second": total_output_tokens / total_seconds,
        "mean_tpot_seconds": total_seconds / args.max_new_tokens,
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None
        ),
        "optimization": "sequential_kv_cache",
    }
    del engine
    cleanup()
    return {"metrics": metrics}


def run_static_batch(args: argparse.Namespace, prompts: list[str]) -> dict[str, object]:
    engine = StaticBatchKVCacheEngine(
        model_name=args.model,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    if args.warmup_new_tokens > 0:
        engine.generate_batch(prompts[: min(2, len(prompts))], max_new_tokens=args.warmup_new_tokens)
        cleanup()
    result = engine.generate_batch(prompts, max_new_tokens=args.max_new_tokens)
    del engine
    cleanup()
    return {"metrics": result.metrics.to_dict()}


def main() -> None:
    args = build_parser().parse_args()
    prompts = make_prompts(args)
    results = {
        "model": args.model,
        "num_requests": args.num_requests,
        "prompt_token_length": args.prompt_token_length,
        "max_new_tokens": args.max_new_tokens,
        "warmup_new_tokens": args.warmup_new_tokens,
        "results": {
            "sequential": run_sequential(args, prompts),
            "static_batch": run_static_batch(args, prompts),
        },
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

