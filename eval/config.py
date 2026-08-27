"""Configuração da suíte de avaliação.

Precedência (do menor para o maior peso):

1. padrões deste arquivo;
2. `eval/eval_config.json`, se existir;
3. variáveis de ambiente `EVAL_<CHAVE>` (e as chaves do `.env`);
4. flags de linha de comando do `run_eval.py`.

Assim, rodar em outra máquina normalmente não exige tocar em código: ou se edita o
JSON, ou se passa uma flag.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = ROOT / "eval" / "eval_config.json"

DEVICE_CHOICES = ("auto", "cuda", "cpu")
QUANTIZATION_CHOICES = ("auto", "4bit", "8bit", "none")
RETRIEVER_CHOICES = ("auto", "openai", "local")
BATTERY_CHOICES = ("model", "pipeline", "e2e")


@dataclass
class EvalConfig:
    """Parâmetros da avaliação. Todos os caminhos são relativos à raiz do repositório."""

    # modelo
    model_id: str = "Grupo-de-Estudos-GPRSW/meditron-7b-finetuned-MedQuAD"
    base_model_id: str = "epfl-llm/meditron-7B"
    cache_dir: Optional[str] = None          # None -> finetuning/cache se existir, senão cache padrão do HF
    device: str = "auto"
    quantization: str = "auto"

    # geração
    samples: int = 10
    max_new_tokens: int = 256
    seed: int = 74                            # a mesma seed do treino (finetuning/training.py)

    # dados e saídas
    medquad_dir: str = "finetuning/MedQuAD"
    output_dir: str = "eval/results"

    # bateria ponta a ponta
    retriever: str = "auto"

    # execução
    batteries: tuple = BATTERY_CHOICES

    # ------------------------------------------------------------------ #
    # caminhos resolvidos
    # ------------------------------------------------------------------ #
    def resolved_cache_dir(self) -> Optional[str]:
        if self.cache_dir:
            return str((ROOT / self.cache_dir).resolve()) if not Path(self.cache_dir).is_absolute() else self.cache_dir
        local = ROOT / "finetuning" / "cache"
        return str(local) if local.is_dir() else None

    def resolved_medquad_dir(self) -> Path:
        p = Path(self.medquad_dir)
        return p if p.is_absolute() else ROOT / p

    def resolved_output_dir(self) -> Path:
        p = Path(self.output_dir)
        return p if p.is_absolute() else ROOT / p

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["batteries"] = list(self.batteries)
        data["cache_dir_resolvido"] = self.resolved_cache_dir()
        return data


# --------------------------------------------------------------------------- #
# Carga em camadas
# --------------------------------------------------------------------------- #

def _coerce(name: str, raw: Any) -> Any:
    """Converte um valor cru (JSON ou variável de ambiente) para o tipo do campo."""
    types = {f.name: f.type for f in fields(EvalConfig)}
    target = types.get(name)
    if target in (int, "int"):
        return int(raw)
    if target in (tuple, "tuple"):
        if isinstance(raw, str):
            return tuple(part.strip() for part in raw.split(",") if part.strip())
        return tuple(raw)
    if raw is None:
        return None
    return str(raw)


def load_config(cli_overrides: Optional[Dict[str, Any]] = None) -> EvalConfig:
    """Monta a configuração aplicando as quatro camadas de precedência."""
    values: Dict[str, Any] = {}

    # camada 2: arquivo JSON opcional
    if CONFIG_FILE.is_file():
        try:
            file_values = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            values.update({k: v for k, v in file_values.items() if k in {f.name for f in fields(EvalConfig)}})
        except json.JSONDecodeError as exc:
            print(f"[aviso] {CONFIG_FILE.name} ignorado: JSON inválido ({exc})", file=sys.stderr)

    # camada 3: variáveis de ambiente EVAL_*
    for f in fields(EvalConfig):
        env_value = os.getenv(f"EVAL_{f.name.upper()}")
        if env_value is not None:
            values[f.name] = env_value

    # camada 4: linha de comando
    if cli_overrides:
        values.update({k: v for k, v in cli_overrides.items() if v is not None})

    coerced = {k: _coerce(k, v) for k, v in values.items()}
    config = EvalConfig(**coerced)
    validate(config)
    return config


def validate(config: EvalConfig) -> None:
    if config.device not in DEVICE_CHOICES:
        raise ValueError(f"device deve ser um de {DEVICE_CHOICES}, veio '{config.device}'")
    if config.quantization not in QUANTIZATION_CHOICES:
        raise ValueError(f"quantization deve ser um de {QUANTIZATION_CHOICES}, veio '{config.quantization}'")
    if config.retriever not in RETRIEVER_CHOICES:
        raise ValueError(f"retriever deve ser um de {RETRIEVER_CHOICES}, veio '{config.retriever}'")
    for battery in config.batteries:
        if battery not in BATTERY_CHOICES:
            raise ValueError(f"bateria desconhecida '{battery}'; use {BATTERY_CHOICES}")
    if config.samples < 1:
        raise ValueError("samples precisa ser >= 1")
    if config.max_new_tokens < 16:
        raise ValueError("max_new_tokens precisa ser >= 16")


# --------------------------------------------------------------------------- #
# Detecção de hardware
# --------------------------------------------------------------------------- #

def hardware_info() -> Dict[str, Any]:
    """Descreve a máquina atual. Nunca lança exceção: campos ausentes viram None."""
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "so": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or None,
        "cuda_disponivel": False,
        "gpu": None,
        "vram_gb": None,
        "torch": None,
        "transformers": None,
        "peft": None,
        "bitsandbytes": None,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_disponivel"] = bool(torch.cuda.is_available())
        if info["cuda_disponivel"]:
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["vram_gb"] = round(props.total_memory / 1e9, 1)
    except Exception:  # noqa: BLE001 -- ambiente sem torch continua utilizável nas outras baterias
        pass
    for mod in ("transformers", "peft", "bitsandbytes"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass
    return info


def resolve_device(config: EvalConfig, hw: Optional[Dict[str, Any]] = None) -> str:
    hw = hw or hardware_info()
    if config.device != "auto":
        return config.device
    return "cuda" if hw["cuda_disponivel"] else "cpu"


def resolve_quantization(config: EvalConfig, device: str, hw: Optional[Dict[str, Any]] = None) -> str:
    """Escolhe a quantização adequada ao hardware quando o modo é `auto`.

    O modelo tem 7B de parâmetros: ~13,5 GB em fp16 e ~4 GB em 4-bit. Em GPU pequena
    a quantização 4-bit (a mesma usada no treino) é o único caminho viável.
    """
    hw = hw or hardware_info()
    if config.quantization != "auto":
        return config.quantization
    if device != "cuda":
        return "none"                      # bitsandbytes exige GPU
    vram = hw.get("vram_gb") or 0
    if vram >= 16:
        return "none"                      # cabe em fp16 sem quantizar
    if not hw.get("bitsandbytes"):
        return "none"                      # sem bitsandbytes não há como quantizar
    return "4bit"


def openai_key_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def resolve_retriever(config: EvalConfig) -> str:
    """`auto` usa o FAISS + OpenAI quando dá, senão cai no retriever local."""
    if config.retriever != "auto":
        return config.retriever
    if openai_key_available() and _modules_available("faiss", "langchain_openai"):
        return "openai"
    return "local"


def _modules_available(*names: str) -> bool:
    import importlib.util

    return all(importlib.util.find_spec(n) is not None for n in names)


def missing_modules(*names: str) -> list:
    import importlib.util

    return [n for n in names if importlib.util.find_spec(n) is None]


def config_from_file(path: str) -> EvalConfig:
    """Reconstrói uma configuração já resolvida (usado quando `run_eval.py` chama as
    baterias em subprocessos)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    known = {f.name for f in fields(EvalConfig)}
    config = EvalConfig(**{k: _coerce(k, v) for k, v in data.items() if k in known})
    validate(config)
    return config


def dump_config(config: EvalConfig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
