# Evidências de CI/CD

As evidências estão anexadas diretamente em [`docs/evidencias/`](../).
Este documento registra o inventário e os resultados observados nos arquivos.

## Identificação da execução

Identificação extraída dos arquivos e das capturas:

| Informação | Valor |
| --- | --- |
| Data registrada no JSON | `2026-09-09T01:38:47`, sem indicação de fuso horário. |
| Branch da entrega | `main`, conforme captura do CD. |
| SHA em `commit.txt` | `c0e2d87fcfeb8bdb65e03afb49cb24724320d0de` |
| Link do commit | [c0e2d87](https://github.com/Grupo-de-Estudos-GPRSW/tech-challenge-3/commit/c0e2d87fcfeb8bdb65e03afb49cb24724320d0de) |
| Execução de CD | #4, merge do PR #7, status `Success`, duração de 30 segundos. |
| Links individuais de CI e CD | Não constam nos arquivos; ainda devem ser registrados pela equipe. |

A captura do CD mostra quatro execuções consecutivas aprovadas e os jobs
`validar / Pipeline sem GPU` e `Preparar entrega` aprovados na execução #4.
O prefixo `c0e2d87` exibido coincide com `commit.txt`.

O JSON registra **11 casos: 9 OK, 2 achados, 0 falhas e 0 erros**, em 0,96 segundo,
usando Python 3.11.16 em Linux 6.17.0-1022-azure, x86_64, sem GPU/CUDA.
Os dois achados dizem respeito às limitações conhecidas do guardrail.

A captura de CI mostra execuções aprovadas na `main` e em pull requests, incluindo
o CI #11 do PR #7. Esse CI de pull request não é a mesma execução da validação
interna do CD #4. 

## Arquivos anexados

| Arquivo | Conteúdo |
| --- | --- |
| [evidencias-ci.png](../evidencias-ci.png) | Histórico e job de CI aprovados. |
| [evidencias-cd.png](../evidencias-cd.png) | Histórico e entrega #4 aprovada, com dois artefatos. |
| [pipeline.json](../pipeline.json) | Resultado da bateria determinística. |
| [python.txt](../python.txt) | Python 3.11.16, consistente com o JSON. |
| [dependencias.txt](../dependencias.txt) | Versões efetivas, incluindo LangGraph 1.2.11 e FastAPI 0.141.1. |
| [commit.txt](../commit.txt) | SHA completo associado à entrega. |
| [SHA256SUMS.txt](../SHA256SUMS.txt) | Checksum registrado para `projeto.zip`. |

O checksum registrado é
`7ccbd33dd1f5e3e7e3184695ddd1f9204423493a2bfe3b576666b68fd24f9104`.
O ZIP não está neste diretório; 

As evidências documentam CI e preparação da entrega. Não demonstram deploy em
servidor nem avaliação do modelo real em GPU.

Consulte a [documentação de CI/CD](../../ci-cd.md) para detalhes do fluxo.
