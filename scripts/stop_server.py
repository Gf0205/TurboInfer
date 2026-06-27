from __future__ import annotations

import argparse
import os
import signal
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stop a background TurboInfer server.")
    parser.add_argument("--pid-file", default="reports/server.pid")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pid_path = Path(args.pid_file)
    if not pid_path.exists():
        print(f"No PID file found: {pid_path}")
        return

    pid = int(pid_path.read_text(encoding="utf-8").strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Stopped TurboInfer server pid={pid}")
    except ProcessLookupError:
        print(f"Process already stopped: pid={pid}")
    finally:
        pid_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

