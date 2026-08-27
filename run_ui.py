"""Entry point for the chat interface.

    python run_ui.py [--host 127.0.0.1] [--port 8000] [--no-browser]

Serves the web UI on http://127.0.0.1:8000 and loads the LangGraph pipeline
(`src/graph.py`) in the background, so the page is usable while the fine-tuned
model is still being downloaded/loaded.
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the GPRSW medical assistant chat UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser window.")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (development).")
    args = parser.parse_args()

    try:
        import uvicorn
    except ModuleNotFoundError:
        print(
            "The web interface needs FastAPI and Uvicorn:\n"
            "    pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    print(f"\n  Grupo-de-Estudos-GPRSW · Medical Assistant\n  {url}\n")

    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run("webapp.server:api", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
