from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()

# The notebook can use the interactive widget, but the web UI has no TTY: when a token
# is available in the environment (or in .env), log in with it instead of prompting.
_hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if _hf_token:
    login(token=_hf_token)
else:
    login()

import torch
from transformers import BitsAndBytesConfig

# Load model and tokenizer to GPU if available
device_map = 'auto'
if torch.cuda.is_available():
    device_map = 'cuda'
    print("Model will load to GPU.")
else:
    print("GPU not available, model running on CPU.")


QUANTIZATION_CHOICES = ("4bit", "8bit", "none")

_QUANTIZATION_HELP = (
    "Escolha como o modelo de 7B deve ser carregado definindo MODEL_QUANTIZATION:\n"
    "  4bit  ~4 GB de VRAM (nf4, a mesma configuração do treino em "
    "finetuning/foundation_model.py)\n"
    "  8bit  ~7 GB de VRAM\n"
    "  none  fp16, ~13,5 GB de VRAM\n"
    "Defina no .env (MODEL_QUANTIZATION=4bit) ou pela flag "
    "`python run_ui.py --quantization 4bit`."
)


def _quantization_config():
    """Quantização escolhida por quem executa a aplicação.

    Não há valor padrão: a escolha é obrigatória e vem de `MODEL_QUANTIZATION`
    (variável de ambiente ou `.env`), que o `run_ui.py` também aceita como
    `--quantization`.

    A decisão é de quem executa porque depende da placa disponível e do
    compromisso entre memória e fidelidade — e porque, em fp16, uma GPU pequena
    chega a "carregar" o modelo (o driver transborda para a memória compartilhada)
    e só falha com out of memory na hora de gerar a primeira resposta, o que torna
    um palpite silencioso pior do que uma escolha explícita.
    """
    modo = os.getenv("MODEL_QUANTIZATION", "").strip().lower()
    if not modo:
        raise ValueError("MODEL_QUANTIZATION não definido.\n" + _QUANTIZATION_HELP)
    if modo not in QUANTIZATION_CHOICES:
        raise ValueError(
            f"MODEL_QUANTIZATION inválido: '{modo}'. Use um de "
            f"{', '.join(QUANTIZATION_CHOICES)}.\n" + _QUANTIZATION_HELP
        )

    if not torch.cuda.is_available():
        if modo != "none":
            print(f"Quantization '{modo}' requires a GPU; running on CPU without quantization.")
        return None

    print(f"Quantization: {modo} (from MODEL_QUANTIZATION).")

    if modo == "none":
        return None
    if modo == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


# Resolvido antes de qualquer download: uma escolha ausente ou inválida falha
# imediatamente, em vez de depois de baixar o tokenizer e os pesos.
quantization_config = _quantization_config()

model_name = "Grupo-de-Estudos-GPRSW/meditron-7b-finetuned-MedQuAD"
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device_map)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    quantization_config=quantization_config,
)
