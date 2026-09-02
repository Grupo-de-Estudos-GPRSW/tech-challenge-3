"""Amostragem determinística do MedQuAD para a bateria de qualidade do modelo.

Varrer os 11.274 XMLs leva minutos, então a amostragem sorteia os arquivos primeiro
(round-robin entre as 12 pastas de origem, para não concentrar tudo em uma fonte) e
só faz o parse dos sorteados. Com a mesma `seed`, a amostra é sempre a mesma.
"""

from __future__ import annotations

import json
import os
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

# Perguntas clínicas do próprio pipeline (mock_patient_db + mock_protocols).
# Não existem no MedQuAD, então servem como sinal fora da distribuição de treino:
# são avaliadas qualitativamente, sem resposta de referência.
CLINICAL_PROBES = [
    {
        "id": "clinico-1",
        "fonte": "pipeline",
        "pergunta": "What is the recommended treatment for a patient with Type 2 Diabetes "
                    "already taking Metformin and Lisinopril?",
        "resposta_referencia": "",
    },
    {
        "id": "clinico-2",
        "fonte": "pipeline",
        "pergunta": "A patient with coronary artery disease is on Aspirin and Atorvastatin "
                    "and has a pending stress test. Should I prescribe a new medication?",
        "resposta_referencia": "",
    },
    {
        "id": "clinico-3",
        "fonte": "pipeline",
        "pergunta": "What follow-up is recommended after pneumonia has resolved?",
        "resposta_referencia": "",
    },
]


def parse_pairs(file_path: Path) -> List[Dict[str, str]]:
    """Pares pergunta/resposta de um XML do MedQuAD (mesma lógica de finetuning/data_loading.py)."""
    try:
        tree = ET.parse(file_path)
    except ET.ParseError:
        return []
    pairs = []
    for qapair in tree.iter("QAPair"):
        question, answer = qapair.find("Question"), qapair.find("Answer")
        if question is None or answer is None:
            continue
        if not question.text or not answer.text:
            continue
        pairs.append({
            "pergunta": question.text.strip(),
            "resposta_referencia": answer.text.strip(),
            "qtype": (question.attrib or {}).get("qtype", ""),
        })
    return pairs


def build_sample(medquad_dir: Path, n: int, seed: int,
                 include_clinical: bool = True) -> Optional[List[Dict[str, str]]]:
    """Sorteia `n` pares do MedQuAD e acrescenta as sondas clínicas.

    Devolve None quando o dataset não está disponível na máquina.
    """
    if not medquad_dir.is_dir():
        return None

    rng = random.Random(seed)
    folders = sorted(f for f in os.listdir(medquad_dir) if (medquad_dir / f).is_dir() and f != ".git")
    if not folders:
        return None

    # ordem de visita: arquivos embaralhados dentro de cada pasta, pastas em round-robin
    queues = []
    for folder in folders:
        files = sorted(f for f in os.listdir(medquad_dir / folder) if f.endswith(".xml"))
        rng.shuffle(files)
        queues.append((folder, files))

    sample: List[Dict[str, str]] = []
    position = 0
    while len(sample) < n and any(files for _, files in queues):
        folder, files = queues[position % len(queues)]
        position += 1
        if not files:
            continue
        pairs = parse_pairs(medquad_dir / folder / files.pop())
        if not pairs:
            continue                       # arquivo sem resposta (removida por licenciamento)
        pair = pairs[rng.randrange(len(pairs))]
        sample.append({
            "id": f"medquad-{len(sample) + 1}",
            "fonte": folder,
            "pergunta": pair["pergunta"],
            "resposta_referencia": pair["resposta_referencia"],
            "qtype": pair["qtype"],
        })

    if include_clinical:
        sample.extend(CLINICAL_PROBES)
    return sample


def load_or_build(medquad_dir: Path, n: int, seed: int, cache_path: Path,
                  include_clinical: bool = True) -> Optional[List[Dict[str, str]]]:
    """Reaproveita `cache_path` quando ele já corresponde aos mesmos parâmetros."""
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("seed") == seed and cached.get("n") == n and \
                    cached.get("com_clinicas") == include_clinical:
                return cached["itens"]
        except (json.JSONDecodeError, KeyError):
            pass

    sample = build_sample(medquad_dir, n, seed, include_clinical)
    if sample is None:
        return None

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(
        {"seed": seed, "n": n, "com_clinicas": include_clinical, "itens": sample},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return sample


if __name__ == "__main__":
    from eval.config import ROOT, load_config

    config = load_config()
    itens = load_or_build(config.resolved_medquad_dir(), config.samples, config.seed,
                          ROOT / "eval" / "data" / "sample.json")
    if itens is None:
        raise SystemExit(f"MedQuAD não encontrado em {config.resolved_medquad_dir()}")
    for item in itens:
        print(f"[{item['id']}] ({item['fonte']}) {item['pergunta'][:90]}")
    print(f"\n{len(itens)} itens gravados em eval/data/sample.json")
