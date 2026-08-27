# Projeto de Chatbot Médico

Este projeto visa ajustar um modelo de IA (finetuning) e construir um pipeline LangChain/LangGraph para criar um chatbot médico para auxiliar médicos.

Para utilizar CUDA, é necessário o torch compilado com suporte a CUDA (uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu132)

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

### Interface web — `run_ui.py` e `webapp/`

Interface de chat para conversar com o modelo ajustado usando exatamente o mesmo grafo
(`src/graph.py`). Cada mensagem enviada dispara uma execução do LangGraph e a UI mostra,
em tempo real, por quais nós o pipeline passou.

```bash
python run_ui.py            # http://127.0.0.1:8000
```

| Módulo | Conteúdo |
| --- | --- |
| `run_ui.py` | Entrada da aplicação (`--host`, `--port`, `--no-browser`, `--reload`). |
| `webapp/pipeline.py` | Carga preguiçosa do grafo em background, sessões e execução de um turno. |
| `webapp/server.py` | API FastAPI: `/api/status`, `/api/chat` (SSE) e `/api/session/reset`. |
| `webapp/static/` | Frontend (HTML/CSS/JS sem build) com o logo do Grupo-de-Estudos-GPRSW. |

Detalhes:

- O servidor sobe imediatamente e carrega o vectorstore e o modelo ajustado em uma thread
  separada; o indicador no topo da página mostra `Carregando`/`Pipeline ready`/erro, com a
  mensagem exata da exceção caso algo falhe.
- A resposta é transmitida por Server-Sent Events: cada nó do grafo concluído vira um passo
  na "Pipeline trace" da mensagem.
- O painel **Sources** de cada resposta traz o registro do paciente (`patient_context`), os
  protocolos recuperados (`retrieved_docs`), o rascunho retido quando o guardrail exige
  aprovação médica e o caminho do log de auditoria.
- `GraphState` é de turno único; a sessão do navegador apenas memoriza o último `patient_id`,
  para que perguntas de acompanhamento continuem no mesmo paciente. O botão **New chat**
  limpa a sessão.
- Login no Hugging Face: como o servidor não tem terminal interativo, defina `HF_TOKEN`
  (no ambiente ou no `.env`) — `src/model_loading.py` usa o token quando ele existe e só
  cai no prompt interativo quando não existe.

### Finetuning — `finetuning/medical_chatbot_finetuning.ipynb` e `finetuning/`

| Módulo | Conteúdo |
| --- | --- |
| `config.py` | Imports, constantes de configuração/hiperparâmetros e login no Hugging Face. |
| `data_loading.py` | Clone do repositório MedQuAD, parse dos XMLs e montagem do `df`. |
| `foundation_model.py` | Quantização 4-bit e carga do `tokenizer` e do `model`. |
| `dataset_formatting.py` | Template Alpaca e montagem do `dataset`. |
| `training.py` | LoRA/PEFT, `SFTTrainer`, treino, `save_model` e `push_to_hub`. |

Esse notebook deve rodar com `finetuning/` como diretório de trabalho, pois os caminhos
(`MedQuAD`, `./cache`, `./meditron-finetuned`) são relativos a ela.

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
- **Token do Hugging Face** (opcional): `HF_TOKEN` no `.env` para login não interativo,
  necessário ao rodar a interface web.
- **FAISS**: `requirements.txt` traz `faiss-cpu`, que tem wheels para Windows; o notebook
  instala `faiss-gpu`, adequado ao ambiente do Colab.
