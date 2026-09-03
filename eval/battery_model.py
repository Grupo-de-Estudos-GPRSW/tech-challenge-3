"""Bateria A — qualidade das respostas do modelo ajustado.

Gera uma resposta para cada item da amostra (perguntas do MedQuAD + sondas clínicas
do pipeline) usando o mesmo template Alpaca do treino, e mede sobreposição com a
resposta de referência, comportamento de parada, repetição, artefatos de fonte e
desempenho.

Uso direto:  python -m eval.battery_model [--samples 10] [--device cpu] ...
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from eval import metrics
from eval.config import ROOT, EvalConfig, config_from_file, hardware_info, load_config
from eval.sample_medquad import load_or_build


def run(config: EvalConfig) -> Dict[str, Any]:
    iniciado = time.perf_counter()
    base = {
        "bateria": "model",
        "executado_em": datetime.datetime.now().isoformat(timespec="seconds"),
        "ambiente": hardware_info(),
        "configuracao": config.to_dict(),
    }

    itens = load_or_build(config.resolved_medquad_dir(), config.samples, config.seed,
                          ROOT / "eval" / "data" / "sample.json")
    if itens is None:
        return {**base, "status": "pulada",
                "motivo": f"MedQuAD não encontrado em {config.resolved_medquad_dir()}",
                "segundos": round(time.perf_counter() - iniciado, 2)}

    from eval.model_runner import ModelRunner

    runner = ModelRunner(config=config)
    try:
        runner.load()
    except Exception as exc:  # noqa: BLE001 -- sem modelo, a bateria é pulada com o motivo
        import traceback

        return {**base, "status": "pulada",
                "motivo": f"falha ao carregar o modelo: {type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc().splitlines()[-8:],
                "segundos": round(time.perf_counter() - iniciado, 2)}

    resultados: List[Dict[str, Any]] = []
    for i, item in enumerate(itens, start=1):
        print(f"  [{i}/{len(itens)}] {item['id']} ({item['fonte']})...", flush=True)
        try:
            geracao = runner.generate(item["pergunta"])
        except Exception as exc:  # noqa: BLE001
            resultados.append({**{k: item[k] for k in ("id", "fonte")},
                               "erro": f"{type(exc).__name__}: {exc}"})
            continue

        avaliacao = metrics.evaluate_generation(geracao.texto, item.get("resposta_referencia", ""))
        resultados.append({
            "id": item["id"],
            "fonte": item["fonte"],
            "qtype": item.get("qtype", ""),
            "pergunta": item["pergunta"],
            "resposta_gerada": geracao.texto,
            "resposta_referencia": item.get("resposta_referencia", ""),
            "tokens_gerados": geracao.tokens_gerados,
            "parou_por_eos": geracao.parou_por_eos,
            "segundos": geracao.segundos,
            "tokens_por_segundo": geracao.tokens_por_segundo,
            **avaliacao,
        })

    modelo = runner.describe()
    runner.unload()

    com_referencia = [r for r in resultados if r.get("resposta_referencia")]
    sem_referencia = [r for r in resultados if not r.get("resposta_referencia") and "erro" not in r]
    validos = [r for r in resultados if "erro" not in r]

    resumo = {
        "itens": len(resultados),
        "com_referencia": len(com_referencia),
        "sondas_clinicas": len(sem_referencia),
        "erros": sum(1 for r in resultados if "erro" in r),
        "rouge_l": metrics.aggregate(com_referencia, "rouge_l"),
        "token_f1": metrics.aggregate(com_referencia, "token_f1"),
        "token_precisao": metrics.aggregate(com_referencia, "token_precisao"),
        "token_recall": metrics.aggregate(com_referencia, "token_recall"),
        "repeticao_trigramas": metrics.aggregate(validos, "repeticao_trigramas"),
        "tokens_gerados": metrics.aggregate(validos, "tokens_gerados"),
        "segundos_por_resposta": metrics.aggregate(validos, "segundos"),
        "tokens_por_segundo": metrics.aggregate(validos, "tokens_por_segundo"),
        "parou_por_eos": _proporcao(validos, lambda r: r["parou_por_eos"]),
        "com_artefato_de_fonte": _proporcao(validos, lambda r: bool(r["artefatos_de_fonte"])),
        "respostas_vazias": _proporcao(validos, lambda r: r["vazia"]),
    }

    return {**base, "status": "ok", "modelo": modelo,
            "segundos": round(time.perf_counter() - iniciado, 2),
            "resumo": resumo, "resultados": resultados}


def _proporcao(linhas: List[Dict[str, Any]], predicado) -> Dict[str, Any]:
    if not linhas:
        return {"quantidade": 0, "total": 0, "proporcao": None}
    quantidade = sum(1 for linha in linhas if predicado(linha))
    return {"quantidade": quantidade, "total": len(linhas),
            "proporcao": round(quantidade / len(linhas), 3)}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Bateria A — qualidade do modelo ajustado")
    parser.add_argument("--effective-config", help="JSON de configuração já resolvida")
    parser.add_argument("--samples", type=int)
    parser.add_argument("--max-new-tokens", type=int, dest="max_new_tokens")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--quantization", choices=("4bit", "8bit", "none"))
    args = parser.parse_args(argv)

    if args.effective_config:
        config = config_from_file(args.effective_config)
    else:
        config = load_config({k: v for k, v in vars(args).items()
                              if k != "effective_config" and v is not None})

    resultado = run(config)
    destino = config.resolved_output_dir() / "model.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")

    if resultado["status"] == "pulada":
        print(f"[bateria model] pulada: {resultado['motivo']}")
    else:
        r = resultado["resumo"]
        print(f"[bateria model] {r['itens']} itens em {resultado['segundos']}s "
              f"({resultado['modelo']['device']}/{resultado['modelo']['quantizacao']})")
        print(f"  ROUGE-L medio ....... {r['rouge_l']['media']}")
        print(f"  F1 de tokens medio .. {r['token_f1']['media']}")
        print(f"  parou por EOS ....... {r['parou_por_eos']['quantidade']}/{r['parou_por_eos']['total']}")
        print(f"  tokens/s medio ...... {r['tokens_por_segundo']['media']}")
    print(f"  -> {destino}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
