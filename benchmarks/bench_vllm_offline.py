from __future__ import annotations

import argparse
import json
import time

from transformers import AutoTokenizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an offline vLLM benchmark.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-requests", type=int, default=8)
    parser.add_argument("--prompt-token-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def make_prompt(tokenizer: AutoTokenizer, target_tokens: int, suffix: str) -> str:
    seed_text = (
        "LLM serving systems use scheduling, batching, KV cache, memory bandwidth, "
        "and latency metrics to balance throughput and tail latency. "
    )
    prompt = f"{suffix}. {seed_text}"
    while len(tokenizer(prompt, return_tensors="pt")["input_ids"][0]) < target_tokens:
        prompt += seed_text
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0][:target_tokens]
    return tokenizer.decode(input_ids, skip_special_tokens=True)


def main() -> None:
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit(
            "vLLM is not installed. Install it on a GPU machine with: pip install vllm"
        ) from exc

    args = build_parser().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    prompts = [
        make_prompt(tokenizer, args.prompt_token_length, suffix=f"Request {idx}")
        for idx in range(args.num_requests)
    ]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )

    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    total_seconds = time.perf_counter() - started

    output_tokens = [len(output.outputs[0].token_ids) for output in outputs]
    total_output_tokens = sum(output_tokens)
    result = {
        "engine": "vllm_offline",
        "model": args.model,
        "num_requests": args.num_requests,
        "prompt_token_length": args.prompt_token_length,
        "max_new_tokens": args.max_new_tokens,
        "total_output_tokens": total_output_tokens,
        "total_seconds": total_seconds,
        "request_throughput_per_second": args.num_requests / total_seconds,
        "token_throughput_per_second": total_output_tokens / total_seconds,
        "mean_output_tokens": total_output_tokens / len(output_tokens) if output_tokens else 0.0,
        "outputs": [
            {
                "index": idx,
                "output_tokens": len(output.outputs[0].token_ids),
                "text_preview": output.outputs[0].text[:160],
            }
            for idx, output in enumerate(outputs)
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

