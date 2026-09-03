"""Entry point for the chat interface.

    python run_ui.py --quantization 4bit|8bit|none
                     [--host 127.0.0.1] [--port 8000] [--no-browser]

Serves the web UI on http://127.0.0.1:8000 and loads the LangGraph pipeline
(`src/graph.py`) in the background, so the page is usable while the fine-tuned
model is still being downloaded/loaded.

The quantization mode is required: pass `--quantization` or set
MODEL_QUANTIZATION in the environment or `.env`.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser

QUANTIZATION_CHOICES = ("4bit", "8bit", "none")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the GPRSW medical assistant chat UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser window.")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (development).")
    parser.add_argument(
        "--quantization", choices=QUANTIZATION_CHOICES,
        help="Required. How to load the fine-tuned model: 4bit (~4 GB of VRAM), "
             "8bit (~7 GB) or none (fp16, ~13.5 GB). May also be set as "
             "MODEL_QUANTIZATION in the environment or .env; the flag wins.",
    )
    args = parser.parse_args()

    # O `.env` precisa estar carregado antes da checagem abaixo: sem isso, quem define
    # MODEL_QUANTIZATION apenas no arquivo seria barrado como se não tivesse escolhido.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ModuleNotFoundError:
        pass

    # Precedência: flag > variável de ambiente / .env. Não há padrão: a escolha é
    # obrigatória. Precisa valer antes de src.model_loading ser importado, o que só
    # acontece na thread de carga iniciada pelo servidor.
    if args.quantization:
        os.environ["MODEL_QUANTIZATION"] = args.quantization

    quantization = os.getenv("MODEL_QUANTIZATION", "").strip().lower()
    if not quantization:
        print(
            "Choose how to load the model: pass --quantization "
            f"{{{'|'.join(QUANTIZATION_CHOICES)}}} or set MODEL_QUANTIZATION in your .env.\n"
            "  4bit  ~4 GB of VRAM (nf4, the same setup used for training)\n"
            "  8bit  ~7 GB of VRAM\n"
            "  none  fp16, ~13.5 GB of VRAM",
            file=sys.stderr,
        )
        return 2
    if quantization not in QUANTIZATION_CHOICES:
        print(
            f"Invalid MODEL_QUANTIZATION: '{quantization}'. "
            f"Use one of {', '.join(QUANTIZATION_CHOICES)}.",
            file=sys.stderr,
        )
        return 2

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
    print(f"\n  Grupo-de-Estudos-GPRSW · Medical Assistant\n  {url}"
          f"\n  model quantization: {quantization}\n")

    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    uvicorn.run("webapp.server:api", host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
