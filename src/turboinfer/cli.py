from __future__ import annotations

import argparse
import json

from turboinfer.engine import KVCacheEngine, NaiveEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TurboInfer baseline generation.")
    parser.add_argument(
        "--engine",
        choices=["naive", "kv-cache"],
        default="naive",
        help="Generation engine to run.",
    )
    parser.add_argument("--model", required=True, help="Hugging Face model name or path.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    engine_cls = KVCacheEngine if args.engine == "kv-cache" else NaiveEngine
    engine = engine_cls(
        model_name=args.model,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    result = engine.generate(args.prompt, max_new_tokens=args.max_new_tokens)

    if args.json:
        print(
            json.dumps(
                {"text": result.text, "metrics": result.metrics.to_dict()},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    print("=== Generated Text ===")
    print(result.text)
    print()
    print("=== Metrics ===")
    for key, value in result.metrics.to_dict().items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
