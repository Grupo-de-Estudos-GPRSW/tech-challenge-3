# Suíte de avaliação

Mede o modelo ajustado e o pipeline LangGraph deste repositório. Foi feita para ser
reexecutada em máquinas diferentes sem edição de código: o hardware é detectado e a
configuração vem em camadas.

```bash
python run_eval.py --quantization 4bit           # roda tudo, adaptando-se ao hardware
python run_eval.py --quantization 4bit --check   # só o diagnóstico, não executa nada
python run_eval.py --quantization 4bit --quick   # 3 perguntas e 128 tokens, valida o caminho
```

A quantização é obrigatória e não tem padrão — `4bit`, `8bit` ou `none`. Todo o resto
continua se adaptando ao hardware.

O resultado final fica em `eval/results/*.json` e alimenta o [RELATORIO.md](../RELATORIO.md).

## As três baterias

| Bateria | O que mede | Precisa de GPU? | Tempo típico |
| --- | --- | --- | --- |
| `pipeline` | Roteamento do grafo, guardrails, gate de validação humana, auditoria, sessões e API do chat — com stubs no lugar do LLM e do vectorstore. | não | segundos |
| `model` | Qualidade das respostas do modelo ajustado em perguntas do MedQuAD e em sondas clínicas: ROUGE-L, F1 de tokens, parada por EOS, repetição, artefatos de fonte, latência. | recomendável | ~10 min em GPU |
| `e2e` | O pipeline completo com o modelo ajustado de verdade: nós percorridos, protocolos recuperados, veredito do guardrail e log de auditoria. | recomendável | ~5 min em GPU |

Rodar uma bateria isolada:

```bash
python run_eval.py --quantization 4bit --only pipeline
python -m eval.battery_model --quantization 4bit --samples 5   # também funciona sozinha
```

## Configuração

Precedência, do menor para o maior peso:

1. padrões em `eval/config.py`;
2. `eval/eval_config.json` (opcional — crie o arquivo para fixar valores da sua máquina);
3. variáveis de ambiente `EVAL_<CHAVE>` (e o `.env` da raiz);
4. flags de linha de comando.

| Chave | Padrão | Efeito |
| --- | --- | --- |
| `model_id` | `Grupo-de-Estudos-GPRSW/meditron-7b-finetuned-MedQuAD` | Repositório do adapter LoRA ajustado. |
| `base_model_id` | `epfl-llm/meditron-7B` | Modelo base sobre o qual o adapter é aplicado. |
| `cache_dir` | `finetuning/cache` se existir | Evita rebaixar os ~13 GB do modelo base. |
| `device` | `auto` | `auto` \| `cuda` \| `cpu`. |
| `quantization` | — | **Obrigatório.** `4bit` \| `8bit` \| `none`. Sem padrão: defina por flag, por `EVAL_QUANTIZATION` ou no JSON. Rebaixado para `none` só quando é impossível aplicar (sem GPU ou sem bitsandbytes), e o rebaixamento é registrado. |
| `samples` | `10` | Perguntas do MedQuAD na bateria `model` (mais 3 sondas clínicas fixas). |
| `max_new_tokens` | `256` | Teto de tokens por resposta. |
| `seed` | `74` | A mesma do treino; a amostra e a geração são determinísticas. |
| `retriever` | `auto` | `openai` (FAISS + embeddings, exige `OPENAI_API_KEY`) \| `local` (BM25 sobre `mock_protocols`) \| `auto`. |
| `medquad_dir` | `finetuning/MedQuAD` | Sem o dataset, a bateria `model` é pulada com o motivo registrado. |
| `output_dir` | `eval/results` | Onde os JSONs são gravados. |

Exemplo de `eval/eval_config.json`:

```json
{
  "samples": 25,
  "max_new_tokens": 384,
  "quantization": "none",
  "retriever": "openai"
}
```

Equivalente por variável de ambiente ou flag:

```bash
EVAL_SAMPLES=25 EVAL_QUANTIZATION=none python run_eval.py
python run_eval.py --samples 25 --quantization none --retriever openai
```

## Adaptação a hardware

`eval/model_runner.py` concentra as decisões:

- aplica a quantização escolhida (4-bit usa a mesma `BitsAndBytesConfig` do treino) e só a
  rebaixa para `none` quando é impossível aplicá-la — sem GPU ou sem bitsandbytes;
- em caso de *out of memory* na geração, corta `max_new_tokens` pela metade e tenta de novo;
- sem GPU, roda em CPU (bem mais lento) — a suíte avisa antes de começar.

Toda degradação é registrada em `modelo.ajustes` no JSON, e cada resultado carrega um
bloco `ambiente` (GPU, VRAM, versões, quantização efetiva) para que execuções de máquinas
diferentes possam ser comparadas.

Cada bateria roda em um subprocesso isolado: uma falta de memória na bateria `model`
não impede a `e2e`, e um pré-requisito ausente vira `status: "pulada"` com o motivo,
nunca um traceback.

## Arquivos

| Arquivo | Papel |
| --- | --- |
| `run_eval.py` (raiz) | Ponto de entrada: diagnóstico, seleção de baterias, resumo. |
| `config.py` | Camadas de configuração e detecção de hardware. |
| `model_runner.py` | Carga do modelo ajustado e geração com o template Alpaca. |
| `metrics.py` | ROUGE-L, F1 de tokens, repetição de trigramas, artefatos de fonte. |
| `sample_medquad.py` | Amostragem determinística e estratificada do MedQuAD. |
| `battery_pipeline.py` / `battery_model.py` / `battery_e2e.py` | As três baterias. |
| `data/sample.json` | Amostra usada (regravada quando `samples` ou `seed` mudam). |
| `results/*.json` | Resultados brutos citados pelo relatório. |
