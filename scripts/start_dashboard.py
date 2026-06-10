"""Dashboard startup script.

Launches the Streamlit dashboard via the streamlit CLI, pointing at the
multi-page app entry point.

Usage:
    python scripts/start_dashboard.py
    python scripts/start_dashboard.py --port 8502
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_APP_PATH = Path("src/observability/dashboard/app.py")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the Smart Knowledge Hub dashboard.")
    parser.add_argument("--port", type=int, default=8501, help="Port to serve on.")
    parser.add_argument("--app", default=str(_APP_PATH), help="Path to the Streamlit app.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not Path(args.app).exists():
        print(f"Dashboard app not found: {args.app}", file=sys.stderr)
        return 1

    cmd = [
        sys.executable, "-m", "streamlit", "run", args.app,
        "--server.port", str(args.port),
    ]
    print(f"Starting dashboard: {' '.join(cmd)}", file=sys.stderr)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
