from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import httpx


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start TurboInfer server in the background.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--preload", action="store_true", help="Send a tiny request to load the model.")
    parser.add_argument("--preload-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--batch-wait-seconds", type=float, default=0.002)
    parser.add_argument("--log-file", default="reports/server.log")
    parser.add_argument("--pid-file", default="reports/server.pid")
    return parser


def wait_for_health(url: str, timeout_seconds: float) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=2.0)
            if response.status_code == 200:
                return
        except Exception as exc:  # noqa: BLE001 - surface final connection issue below
            last_error = exc
        time.sleep(1.0)
    raise TimeoutError(f"server did not become healthy at {url}; last_error={last_error}")


def preload_model(base_url: str, model: str, timeout_seconds: float) -> None:
    url = f"{base_url}/v1/completions"
    payload = {
        "model": model,
        "prompt": "warmup",
        "max_tokens": 1,
        "engine": "kv-cache",
        "temperature": 0.0,
    }
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(url, json=payload)
        response.raise_for_status()


def main() -> None:
    args = build_parser().parse_args()
    log_path = Path(args.log_file)
    pid_path = Path(args.pid_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "turboinfer.server",
        "--model",
        args.model,
        "--device",
        args.device,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--max-batch-size",
        str(args.max_batch_size),
        "--batch-wait-seconds",
        str(args.batch_wait_seconds),
    ]
    if args.trust_remote_code:
        command.append("--trust-remote-code")

    log_file = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pid_path.write_text(str(process.pid), encoding="utf-8")

    health_url = f"http://{args.host}:{args.port}/health"
    try:
        wait_for_health(health_url, args.timeout_seconds)
        if args.preload:
            print("Preloading model with a 1-token warmup request...")
            preload_model(
                base_url=f"http://{args.host}:{args.port}",
                model=args.model,
                timeout_seconds=args.preload_timeout_seconds,
            )
    except Exception:
        process.terminate()
        raise

    print(f"TurboInfer server started: pid={process.pid}")
    print(f"Health: {health_url}")
    print(f"Log: {log_path}")
    print(f"PID file: {pid_path}")


if __name__ == "__main__":
    main()
