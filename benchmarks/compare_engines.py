from __future__ import annotations

import argparse
import gc
import json

import torch
from transformers import AutoTokenizer

from turboinfer.engine import KVCacheEngine, NaiveEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare TurboInfer generation engines.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-token-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def make_prompt(model_name: str, target_tokens: int, trust_remote_code: bool) -> str:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    seed_text = (
        "LLM inference systems use prefill, decode, attention, memory bandwidth, "
        "and KV cache to balance latency and throughput. "
    )
    prompt = seed_text
    while len(tokenizer(prompt, return_tensors="pt")["input_ids"][0]) < target_tokens:
        prompt += seed_text
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0][:target_tokens]
    return tokenizer.decode(input_ids, skip_special_tokens=True)


def run_engine(engine_name: str, args: argparse.Namespace, prompt: str) -> dict[str, object]:
    engine_cls = KVCacheEngine if engine_name == "kv-cache" else NaiveEngine
    engine = engine_cls(
        model_name=args.model,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    result = engine.generate(prompt, max_new_tokens=args.max_new_tokens)
    output = {"text": result.text, "metrics": result.metrics.to_dict()}
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def main() -> None:
    args = build_parser().parse_args()
    prompt = make_prompt(
        model_name=args.model,
        target_tokens=args.prompt_token_length,
        trust_remote_code=args.trust_remote_code,
    )

    results = {
        "model": args.model,
        "prompt_token_length": args.prompt_token_length,
        "max_new_tokens": args.max_new_tokens,
        "results": {
            "naive": run_engine("naive", args, prompt),
            "kv_cache": run_engine("kv-cache", args, prompt),
        },
    }
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
