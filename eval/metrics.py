"""Métricas de qualidade das respostas, em Python puro (sem dependências novas).

As métricas são propositalmente simples e auditáveis: qualquer número do relatório
pode ser reconferido lendo estas funções.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

# Marcas de origem do MedQuAD: as respostas do dataset vêm de páginas do NIH e
# frequentemente terminam remetendo o leitor ao site da fonte. Quando o modelo
# ajustado reproduz esse padrão, está copiando o estilo do dado de treino.
SOURCE_ARTIFACT_PATTERNS = [
    r"\bGARD\b",
    r"\bvisit\s+(?:the\s+)?(?:GARD|MedlinePlus|NIH|NINDS|NIDDK|CDC)\b",
    r"For more information[,\s]",
    r"\bthis (?:web ?site|page)\b",
    r"\bgenetics home reference\b",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall((text or "").lower())


# --------------------------------------------------------------------------- #
# Sobreposição com a resposta de referência
# --------------------------------------------------------------------------- #

def token_f1(prediction: str, reference: str) -> Dict[str, float]:
    """Precisão/recall/F1 sobre o multiconjunto de tokens (estilo SQuAD)."""
    pred, ref = tokenize(prediction), tokenize(reference)
    if not pred or not ref:
        return {"precisao": 0.0, "recall": 0.0, "f1": 0.0}

    from collections import Counter

    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return {"precisao": 0.0, "recall": 0.0, "f1": 0.0}
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return {
        "precisao": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4),
    }


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Maior subsequência comum, em O(len(a) * len(b)) tempo e O(len(b)) memória."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b):
            if token_a == token_b:
                current.append(previous[j] + 1)
            else:
                current.append(max(current[j], previous[j + 1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str, max_tokens: int = 600) -> float:
    """ROUGE-L F1. As sequências são truncadas para manter o custo previsível."""
    pred = tokenize(prediction)[:max_tokens]
    ref = tokenize(reference)[:max_tokens]
    if not pred or not ref:
        return 0.0
    lcs = _lcs_length(pred, ref)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    return round(2 * precision * recall / (precision + recall), 4)


# --------------------------------------------------------------------------- #
# Qualidade intrínseca do texto gerado
# --------------------------------------------------------------------------- #

def trigram_repetition(text: str) -> float:
    """Fração de trigramas repetidos — indicador clássico de degeneração.

    0.0 = nenhum trigrama se repete; valores acima de ~0.3 indicam texto em loop.
    """
    tokens = tokenize(text)
    if len(tokens) < 3:
        return 0.0
    trigrams = [tuple(tokens[i:i + 3]) for i in range(len(tokens) - 2)]
    return round(1 - len(set(trigrams)) / len(trigrams), 4)


def source_artifacts(text: str) -> List[str]:
    """Marcas do MedQuAD encontradas no texto (ver SOURCE_ARTIFACT_PATTERNS)."""
    found = []
    for pattern in SOURCE_ARTIFACT_PATTERNS:
        match = re.search(pattern, text or "", flags=re.IGNORECASE)
        if match:
            found.append(match.group(0).strip())
    return found


def is_empty(text: str) -> bool:
    return not tokenize(text)


def evaluate_generation(prediction: str, reference: str = "") -> Dict[str, object]:
    """Reúne todas as métricas de uma geração.

    Sem referência (perguntas clínicas do pipeline), só as métricas intrínsecas
    são calculadas e as de sobreposição vêm como None.
    """
    result: Dict[str, object] = {
        "tokens_palavra": len(tokenize(prediction)),
        "caracteres": len(prediction or ""),
        "repeticao_trigramas": trigram_repetition(prediction),
        "artefatos_de_fonte": source_artifacts(prediction),
        "vazia": is_empty(prediction),
    }
    if reference:
        result["rouge_l"] = rouge_l(prediction, reference)
        result.update({f"token_{k}": v for k, v in token_f1(prediction, reference).items()})
    else:
        result["rouge_l"] = None
        result["token_f1"] = None
    return result


def aggregate(rows: List[Dict[str, object]], key: str) -> Dict[str, float]:
    """Média/mínimo/máximo de uma métrica numérica, ignorando valores ausentes."""
    values = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    if not values:
        return {"media": None, "min": None, "max": None, "n": 0}
    return {
        "media": round(sum(values) / len(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "n": len(values),
    }
