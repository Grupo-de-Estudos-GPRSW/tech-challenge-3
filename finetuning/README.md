# Fine tuning do chatbot médico

Esta pasta contém o ajuste fino (fine tuning) do modelo que o pipeline em `src/` consome.
Partimos de um foundation model médico aberto, o **`epfl-llm/meditron-7B`**, e o
especializamos em perguntas e respostas clínicas usando o dataset **MedQuAD**. O resultado é
publicado no Hugging Face como **`Grupo-de-Estudos-GPRSW/meditron-7b-finetuned-MedQuAD`**.

O ponto de entrada é o notebook `medical_chatbot_finetuning.ipynb`; todo o código dele foi
movido, sem alterações, para os módulos `.py` desta pasta. Cada módulo importa os anteriores,
então executar as células na ordem reproduz o fluxo original.

> O notebook e os módulos devem rodar com `finetuning/` como diretório de trabalho, pois os
> caminhos (`MedQuAD`, `./cache`, `./meditron-finetuned`) são relativos a ela.

## Módulos

| Módulo | Conteúdo |
| --- | --- |
| `config.py` | Imports, constantes de configuração/hiperparâmetros e login no Hugging Face. |
| `data_loading.py` | Clone do repositório MedQuAD, parse dos XMLs e montagem do `df`. |
| `foundation_model.py` | Quantização 4-bit e carga do `tokenizer` e do `model`. |
| `dataset_formatting.py` | Template Alpaca e montagem do `dataset`. |
| `training.py` | LoRA/PEFT, `SFTTrainer`, treino, `save_model` e `push_to_hub`. |

## Etapas do ajuste

### 1. Configuração (`config.py`)

Define o modelo base, a origem dos dados e os hiperparâmetros, todos calibrados para caber em
uma **RTX 3060 de 12 GB**:

| Constante | Valor | Papel |
| --- | --- | --- |
| `MODEL_ID` | `epfl-llm/meditron-7B` | Foundation model, já pré-treinado em literatura médica. |
| `GITHUB_REPO_ID` | `abachaa/MedQuAD` | Dataset de perguntas e respostas médicas. |
| `BATCH_SIZE` | 4 | Amostras por passo em cada GPU. |
| `GRADIENT_ACCUMULATION_STEPS` | 4 | Batch efetivo de 16 sem estourar a VRAM. |
| `LEARNING_RATE` | 2e-4 | Taxa de aprendizado típica para LoRA. |
| `LR_SCHEDULER_TYPE` | `linear` | Decaimento linear da taxa de aprendizado. |
| `MAX_SEQ_LENGTH` | 512 | Tamanho máximo da sequência de treino. |

O `WANDB_DISABLED` desliga o Weights & Biases, e `login()` autentica no Hugging Face — tanto
para baixar o Meditron quanto para publicar o modelo ajustado ao final.

### 2. Dados (`data_loading.py`)

O repositório [MedQuAD](https://github.com/abachaa/MedQuAD) é clonado na primeira execução
(pulado se a pasta `MedQuAD/` já existir). O MedQuAD reúne pares de pergunta e resposta
extraídos de sites do NIH, organizados em arquivos XML por fonte.

O parse percorre cada `QAPair` dos XMLs e guarda apenas os pares em que **pergunta e resposta
existem e não são vazias** — parte do dataset original tem respostas removidas por questões
de licenciamento. O resultado é um `DataFrame` com as colunas `question` e `answer`.

### 3. Modelo base quantizado (`foundation_model.py`)

Um modelo de 7 bilhões de parâmetros não cabe em 12 GB de VRAM em precisão cheia, então ele é
carregado com quantização **4-bit** via `BitsAndBytesConfig`:

- `load_in_4bit=True` — pesos em 4 bits;
- `bnb_4bit_quant_type="nf4"` — quantização NormalFloat4, melhor para pesos com distribuição normal;
- `bnb_4bit_compute_dtype=torch.float16` — as contas continuam em fp16;
- `bnb_4bit_use_double_quant=True` — quantiza também as constantes de quantização, economizando mais memória.

Os downloads ficam em `./cache` (ignorado pelo git).

### 4. Formatação do dataset (`dataset_formatting.py`)

Cada par pergunta/resposta é convertido em um único texto no template inspirado no model card
do Meditron:

```
###System:
You are a helpful, respectful, and honest assistant. ...

### User:
{question}

### Assistant:
{answer}<eos>
```

O `eos_token` no fim da resposta ensina o modelo a **parar de gerar** no lugar certo. As
colunas originais são removidas no `map`, restando apenas a coluna `text` consumida pelo
treinador. Esse mesmo template é reproduzido em `src/generation_nodes.py` na hora da
inferência — é por isso que o pipeline corta a saída no marcador `### Assistant:`.

### 5. Treino com LoRA (`training.py`)

Em vez de atualizar os 7 bilhões de parâmetros, aplicamos **LoRA** (Low-Rank Adaptation): os
pesos originais ficam congelados e treinamos apenas pequenas matrizes de baixo posto
injetadas nas camadas de atenção.

- `prepare_model_for_kbit_training` — prepara o modelo quantizado para receber gradientes.
- `LoraConfig(r=8, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"])` — adaptadores nas projeções de query, key, value e output da atenção.
- `print_trainable_parameters()` — mostra que apenas uma fração mínima dos parâmetros é treinada.

O `SFTConfig`/`SFTTrainer` (TRL) cuidam do laço de treino supervisionado: 3 épocas,
`optim="adamw_8bit"` (estados do otimizador em 8 bits, mais economia de VRAM), checkpoint a
cada 60 passos em `./meditron-finetuned`, `packing=False` e `seed=74` para reprodutibilidade.

Ao final, `save_model` grava os adaptadores localmente e `push_to_hub` publica o modelo na
organização do grupo no Hugging Face.

## Como executar

```bash
pip install -r ../requirements.txt
cd finetuning
jupyter lab medical_chatbot_finetuning.ipynb
```

Ou, fora do notebook (a partir de `finetuning/`):

```bash
python -c "import training"   # config -> dados -> modelo -> dataset -> treino
```

Pré-requisitos:

- **GPU NVIDIA com CUDA** e `torch` compilado com suporte a CUDA — o `bitsandbytes` e o
  `device_map="cuda"` de `foundation_model.py` não funcionam apenas com CPU.
- **Conta no Hugging Face** com acesso ao `epfl-llm/meditron-7B` e permissão de escrita na
  organização de destino (`login()` em `config.py`; defina `HF_TOKEN` no `.env` para um login
  não interativo).
- Espaço em disco para o cache do modelo base (`./cache`) e para os checkpoints
  (`./meditron-finetuned`). Ambos são ignorados pelo git, assim como a cópia do `MedQuAD`.

## Como o resultado é usado

O modelo publicado é carregado por `src/model_loading.py` e usado pelo nó
`generate_llm_response` do grafo LangGraph — tanto no `example_notebook.ipynb` quanto na
interface de chat (`python run_ui.py`). Ver o [README da raiz](../README.md).
