"""Suíte de avaliação do chatbot médico — ponto de entrada único.

    python run_eval.py                  # roda tudo, adaptando-se ao hardware
    python run_eval.py --check          # só o diagnóstico do ambiente
    python run_eval.py --quick          # amostra pequena, para validar o caminho
    python run_eval.py --only pipeline  # escolhe as baterias
    python run_eval.py --samples 25 --device cpu --quantization none

Cada bateria roda em um subprocesso isolado: uma falha de memória ou de importação
em uma delas não derruba as demais, e cada uma pode ser executada sozinha
(`python -m eval.battery_pipeline`, `-m eval.battery_model`, `-m eval.battery_e2e`).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from eval.config import (BATTERY_CHOICES, ROOT, EvalConfig, dump_config, hardware_info,
                         load_config, missing_modules, openai_key_available, resolve_device,
                         resolve_quantization, resolve_retriever)

BATERIAS = {
    "pipeline": ("eval.battery_pipeline", "pipeline determinístico do grafo (sem LLM, segundos)"),
    "model": ("eval.battery_model", "qualidade das respostas do modelo ajustado"),
    "e2e": ("eval.battery_e2e", "pipeline completo com o modelo ajustado"),
}

# A ordem importa: a bateria barata roda primeiro e o modelo é carregado só uma vez
# por subprocesso das baterias caras.
ORDEM = ("pipeline", "model", "e2e")


# --------------------------------------------------------------------------- #
# Diagnóstico
# --------------------------------------------------------------------------- #

def diagnosticar(config: EvalConfig) -> Dict[str, Any]:
    hw = hardware_info()
    device = resolve_device(config, hw)
    quantization = resolve_quantization(config, device, hw)
    retriever = resolve_retriever(config)

    avisos: List[str] = []
    if not hw["cuda_disponivel"]:
        avisos.append("sem GPU: as baterias 'model' e 'e2e' vão rodar em CPU e podem levar horas")
    elif quantization == "none" and (hw.get("vram_gb") or 0) < 16:
        avisos.append(f"quantização desligada com apenas {hw.get('vram_gb')} GB de VRAM: "
                      "o modelo de 7B provavelmente não vai caber")
    if not config.resolved_medquad_dir().is_dir():
        avisos.append(f"MedQuAD não encontrado em {config.resolved_medquad_dir()}: "
                      "a bateria 'model' será pulada")
    if retriever == "local":
        motivo = "OPENAI_API_KEY ausente" if not openai_key_available() \
            else f"módulos ausentes: {missing_modules('faiss', 'langchain_openai')}"
        avisos.append(f"retriever local (BM25) na bateria 'e2e' — {motivo}")
    if not config.resolved_cache_dir():
        avisos.append("cache local do modelo base não encontrado: a primeira execução "
                      "vai baixar ~13 GB do Hugging Face")

    return {"hardware": hw, "device": device, "quantizacao": quantization,
            "retriever": retriever, "avisos": avisos}


def imprimir_diagnostico(config: EvalConfig, diag: Dict[str, Any], baterias: List[str]) -> None:
    hw = diag["hardware"]
    print("\n=== Ambiente ===")
    print(f"  SO ............. {hw['so']}  |  Python {hw['python']}")
    gpu = hw["gpu"] or "nenhuma"
    if hw.get("vram_gb"):
        gpu += f" ({hw['vram_gb']} GB)"
    print(f"  GPU ............ {gpu}")
    print(f"  torch .......... {hw['torch']}  |  transformers {hw['transformers']}  |  peft {hw['peft']}")
    print(f"  bitsandbytes ... {hw['bitsandbytes'] or 'ausente'}")
    print("\n=== Plano ===")
    print(f"  device ......... {diag['device']}")
    print(f"  quantizacao .... {diag['quantizacao']}")
    print(f"  retriever ...... {diag['retriever']}")
    print(f"  amostra ........ {config.samples} perguntas + 3 sondas clínicas")
    print(f"  max_new_tokens . {config.max_new_tokens}")
    print(f"  saída .......... {config.resolved_output_dir()}")
    print("  baterias:")
    for nome in baterias:
        print(f"    - {nome}: {BATERIAS[nome][1]}")
    if diag["avisos"]:
        print("\n=== Avisos ===")
        for aviso in diag["avisos"]:
            print(f"  ! {aviso}")
    print()


# --------------------------------------------------------------------------- #
# Execução
# --------------------------------------------------------------------------- #

def rodar_bateria(nome: str, config_path: Path) -> Dict[str, Any]:
    modulo = BATERIAS[nome][0]
    print(f"--- bateria '{nome}' ---", flush=True)
    comeco = time.perf_counter()
    processo = subprocess.run(
        [sys.executable, "-m", modulo, "--effective-config", str(config_path)],
        cwd=str(ROOT),
    )
    return {"bateria": nome, "codigo_de_saida": processo.returncode,
            "segundos": round(time.perf_counter() - comeco, 1)}


def resumir(config: EvalConfig, execucoes: List[Dict[str, Any]]) -> None:
    print("\n=== Resumo ===")
    for execucao in execucoes:
        arquivo = config.resolved_output_dir() / f"{execucao['bateria']}.json"
        linha = f"  {execucao['bateria']:<9} {execucao['segundos']:>7.1f}s"
        if not arquivo.is_file():
            print(f"{linha}  (sem arquivo de resultado)")
            continue
        dados = json.loads(arquivo.read_text(encoding="utf-8"))
        if dados.get("status") == "pulada":
            print(f"{linha}  PULADA: {dados['motivo']}")
        elif execucao["bateria"] == "pipeline":
            r = dados["resumo"]
            print(f"{linha}  {r['ok']}/{r['total']} ok, {r['achados']} achado(s), "
                  f"{r['falhas']} falha(s), {r['erros']} erro(s)")
        elif execucao["bateria"] == "model":
            r = dados["resumo"]
            print(f"{linha}  ROUGE-L {r['rouge_l']['media']} | F1 {r['token_f1']['media']} | "
                  f"EOS {r['parou_por_eos']['quantidade']}/{r['parou_por_eos']['total']} | "
                  f"{r['tokens_por_segundo']['media']} tok/s")
        elif execucao["bateria"] == "e2e":
            r = dados["resumo"]
            print(f"{linha}  {r['consultas']} consultas, guardrail em "
                  f"{r['com_guardrail_disparado']}, logs {r['logs_gravados']} "
                  f"(retriever {dados.get('retriever')})")
    print(f"\n  resultados em {config.resolved_output_dir()}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Suíte de avaliação do chatbot médico",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="só imprime o diagnóstico do ambiente e sai")
    parser.add_argument("--quick", action="store_true",
                        help="amostra e geração reduzidas (3 perguntas, 128 tokens)")
    parser.add_argument("--only", nargs="+", choices=BATTERY_CHOICES, metavar="BATERIA",
                        help=f"baterias a executar ({', '.join(BATTERY_CHOICES)})")
    parser.add_argument("--samples", type=int, help="perguntas do MedQuAD na bateria 'model'")
    parser.add_argument("--max-new-tokens", type=int, dest="max_new_tokens")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--quantization", choices=("auto", "4bit", "8bit", "none"))
    parser.add_argument("--retriever", choices=("auto", "openai", "local"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", dest="output_dir")
    args = parser.parse_args(argv)

    overrides = {k: v for k, v in vars(args).items()
                 if k not in ("check", "quick", "only") and v is not None}
    if args.quick:
        overrides.setdefault("samples", 3)
        overrides.setdefault("max_new_tokens", 128)
    if args.only:
        overrides["batteries"] = tuple(args.only)

    try:
        config = load_config(overrides)
    except ValueError as exc:
        print(f"configuração inválida: {exc}", file=sys.stderr)
        return 2

    baterias = [nome for nome in ORDEM if nome in config.batteries]
    diag = diagnosticar(config)
    imprimir_diagnostico(config, diag, baterias)

    if args.check:
        return 0

    config.resolved_output_dir().mkdir(parents=True, exist_ok=True)
    config_path = dump_config(config, config.resolved_output_dir() / "_configuracao_efetiva.json")

    execucoes = [rodar_bateria(nome, config_path) for nome in baterias]
    resumir(config, execucoes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
