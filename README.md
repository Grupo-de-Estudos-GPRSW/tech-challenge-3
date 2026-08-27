# Projeto de Chatbot Médico

Este projeto ajusta um modelo de IA (fine tuning) e constrói um pipeline LangChain/LangGraph
para criar um chatbot médico que auxilia profissionais de saúde. São três partes:

1. **Fine tuning** (`finetuning/`) — especializa o `epfl-llm/meditron-7B` no dataset MedQuAD e
   publica o resultado como `Grupo-de-Estudos-GPRSW/meditron-7b-finetuned-MedQuAD`.
   Ver o [README do fine tuning](finetuning/README.md).
2. **Pipeline** (`src/` + `example_notebook.ipynb`) — o grafo LangGraph que combina o
   prontuário do paciente, os protocolos clínicos internos, o modelo ajustado, os guardrails
   de segurança e a trilha de auditoria.
3. **Interface web** (`webapp/` + `run_ui.py`) — um chat que conversa com o modelo ajustado
   através desse mesmo grafo.

A avaliação do modelo e do pipeline está no **[RELATORIO.md](RELATORIO.md)**, gerado a partir
da suíte de testes em [`eval/`](eval/README.md) (`python run_eval.py`).

## Início rápido

```bash
pip install -r requirements.txt

# .env na raiz do repositório
#   OPENAI_API_KEY=...     embeddings dos protocolos
#   HF_TOKEN=...           login não interativo no Hugging Face

python run_ui.py          # abre o chat em http://127.0.0.1:8000
```

Para utilizar CUDA, é necessário o torch compilado com suporte a CUDA
(`uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu132`).

## Estrutura

Os notebooks continuam sendo o ponto de entrada; todo o código deles foi movido, sem
alterações, para módulos `.py`. Cada módulo importa os anteriores, então executar as
células do notebook na ordem reproduz o fluxo original.

### Pipeline — `example_notebook.ipynb` (raiz) e `src/`

| Módulo | Conteúdo |
| --- | --- |
| `model_loading.py` | Login no Hugging Face e carga do `tokenizer` e do `model` ajustado. |
| `graph_state.py` | `GraphState`, o esquema de estado compartilhado entre os nós. |
| `patient_database.py` | `mock_patient_db`, os registros de pacientes. |
| `protocols_database.py` | `mock_protocols`, os protocolos clínicos internos. |
| `vectorstore.py` | Embeddings da OpenAI, índice FAISS e `retriever`. |
| `input_nodes.py` | `parse_and_validate_input` e `request_patient_id`. |
| `ehr_nodes.py` | `fetch_ehr_context`. |
| `retrieval_nodes.py` | `retrieve_protocols`. |
| `generation_nodes.py` | `generate_llm_response`. |
| `guardrail_nodes.py` | `guardrail_evaluator` e `human_validation_gate`. |
| `audit_nodes.py` | `audit_logger`. |
| `graph.py` | Montagem e compilação do `StateGraph` (`app`). |

O notebook deve rodar com a raiz do repositório como diretório de trabalho, pois os
módulos são importados como `from src.<modulo> import ...`.

Para executar o pipeline fora do notebook:

```python
from src.graph import app

output_state = app.invoke({
    "user_query": "What is the recommended treatment for P001's diabetes?",
    "patient_id": None,
    "missing_patient_id": False,
    "patient_context": {},
    "retrieved_docs": [],
    "llm_output": "",
    "requires_human_approval": False,
    "audit_payload": {}
})
print(output_state["llm_output"])
```

O modelo foi ajustado com perguntas e respostas em inglês (MedQuAD), então as consultas
devem ser feitas em inglês.

### Interface web — `run_ui.py` e `webapp/`

Interface de chat para conversar com o modelo ajustado usando exatamente o mesmo grafo
(`src/graph.py`). Cada mensagem enviada dispara uma execução do LangGraph e a interface
mostra, em tempo real, por quais nós o pipeline passou.

```bash
python run_ui.py            # http://127.0.0.1:8000
```

Opções: `--host`, `--port`, `--no-browser` (não abrir o navegador) e `--reload` (recarregar
ao alterar o código, útil no desenvolvimento).

| Módulo | Conteúdo |
| --- | --- |
| `run_ui.py` | Entrada da aplicação. |
| `webapp/pipeline.py` | Carga preguiçosa do grafo em background, sessões e execução de um turno. |
| `webapp/server.py` | API FastAPI: `/api/status`, `/api/chat` (SSE) e `/api/session/reset`. |
| `webapp/static/` | Frontend (HTML/CSS/JS, sem build) com o logo do Grupo-de-Estudos-GPRSW. |

Detalhes:

- O servidor sobe imediatamente e carrega o índice vetorial e o modelo ajustado em uma
  thread separada; o indicador no topo da página mostra o progresso (`Pipeline ready` quando
  termina) e, se algo falhar, a mensagem exata da exceção.
- A resposta é transmitida por Server-Sent Events: cada nó concluído do grafo vira um passo
  no bloco *Pipeline trace* da mensagem.
- O painel *Sources* de cada resposta traz o registro do paciente (`patient_context`), os
  protocolos recuperados (`retrieved_docs`), o rascunho retido quando o guardrail exige
  aprovação médica e o caminho do log de auditoria.
- O `GraphState` é de turno único; a sessão do navegador apenas memoriza o último
  `patient_id`, para que perguntas de acompanhamento continuem no mesmo paciente. O botão
  *New chat* limpa a sessão.
- Login no Hugging Face: como o servidor não tem terminal interativo, defina `HF_TOKEN`
  (no ambiente ou no `.env`) — `src/model_loading.py` usa o token quando ele existe e só cai
  no prompt interativo quando não existe.
- A interface é servida em inglês, acompanhando o idioma do modelo ajustado.

### Fine tuning — `finetuning/`

O ajuste do modelo (dataset MedQuAD, quantização 4-bit, LoRA/PEFT e publicação no Hugging
Face) está documentado em detalhe no **[README do fine tuning](finetuning/README.md)**.

| Módulo | Conteúdo |
| --- | --- |
| `config.py` | Imports, constantes de configuração/hiperparâmetros e login no Hugging Face. |
| `data_loading.py` | Clone do repositório MedQuAD, parse dos XMLs e montagem do `df`. |
| `foundation_model.py` | Quantização 4-bit e carga do `tokenizer` e do `model`. |
| `dataset_formatting.py` | Template Alpaca e montagem do `dataset`. |
| `training.py` | LoRA/PEFT, `SFTTrainer`, treino, `save_model` e `push_to_hub`. |

O notebook `finetuning/medical_chatbot_finetuning.ipynb` deve rodar com `finetuning/` como
diretório de trabalho, pois os caminhos (`MedQuAD`, `./cache`, `./meditron-finetuned`) são
relativos a ela.

### Avaliação — `run_eval.py` e `eval/`

Suíte que mede o modelo ajustado e o pipeline, com detecção automática de hardware:

```bash
python run_eval.py --check    # diagnóstico do ambiente, sem executar nada
python run_eval.py            # as três baterias
```

São três baterias: `pipeline` (grafo determinístico, sem LLM, segundos), `model` (qualidade
das respostas do modelo ajustado no MedQuAD) e `e2e` (pipeline completo com o modelo real).
Os resultados vão para `eval/results/*.json` e embasam o [RELATORIO.md](RELATORIO.md).
Detalhes de configuração em [`eval/README.md`](eval/README.md).

### Outros diretórios

- `execution_log/`: trilhas de auditoria geradas a cada execução (ignorado pelo git).

## Fluxo do pipeline

```
parse_and_validate_input ──(sem patient_id)──> request_patient_id ──> audit_logger ──> END
            │
     (com patient_id)
            ▼
   fetch_ehr_context ──> retrieve_protocols ──> generate_llm_response ──> guardrail_evaluator
                                                                              │
                                              (conteúdo sensível) ────────────┤
                                                                              ▼
                                                        human_validation_gate ──> audit_logger ──> END
```

## Requisitos

```bash
pip install -r requirements.txt
```

- **Python 3.11+**.
- **Modelo ajustado**: `Grupo-de-Estudos-GPRSW/meditron-7b-finetuned-MedQuAD`, baixado do
  Hugging Face na primeira execução (`login()` é chamado por `src/model_loading.py`).
- **Chave da OpenAI**: `OPENAI_API_KEY` no `.env`, usada em `src/vectorstore.py` para gerar os
  embeddings dos protocolos.
- **Token do Hugging Face**: `HF_TOKEN` no `.env` para login não interativo — obrigatório ao
  rodar a interface web, opcional no notebook.
- **FAISS**: `requirements.txt` traz `faiss-cpu`, que tem wheels para Windows; o notebook
  instala `faiss-gpu`, adequado ao ambiente do Colab.
- **GPU**: opcional para o pipeline (o modelo roda em CPU, porém lentamente) e praticamente
  obrigatória para o fine tuning.
