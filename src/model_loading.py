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


QUANTIZATION_CHOICES = ("auto", "4bit", "8bit", "none")


def _quantization_config():
    """Quantização escolhida por quem executa a aplicação.

    A escolha vem de `MODEL_QUANTIZATION` (variável de ambiente ou `.env`), que o
    `run_ui.py` também aceita como `--quantization`:

    - `4bit`  ~4 GB de VRAM (nf4, a mesma configuração do treino em
              `finetuning/foundation_model.py`);
    - `8bit`  ~7 GB de VRAM;
    - `none`  fp16, ~13,5 GB de VRAM;
    - `auto`  (padrão) decide pela VRAM da placa: `none` a partir de 16 GB, senão `4bit`.

    O padrão é `auto` porque, em fp16, uma GPU pequena chega a "carregar" o modelo — o
    driver transborda para a memória compartilhada — e só falha com out of memory na hora
    de gerar a primeira resposta.
    """
    modo = os.getenv("MODEL_QUANTIZATION", "auto").strip().lower() or "auto"
    if modo not in QUANTIZATION_CHOICES:
        raise ValueError(
            f"MODEL_QUANTIZATION inválido: '{modo}'. Use um de {', '.join(QUANTIZATION_CHOICES)}."
        )

    if not torch.cuda.is_available():
        if modo not in ("auto", "none"):
            print(f"Quantization '{modo}' requires a GPU; running on CPU without quantization.")
        return None

    if modo == "auto":
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        modo = "none" if vram_gb >= 16 else "4bit"
        print(f"Quantization 'auto': {vram_gb:.1f} GB of VRAM -> {modo} "
              "(set MODEL_QUANTIZATION to choose it yourself).")
    else:
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


model_name = "Grupo-de-Estudos-GPRSW/meditron-7b-finetuned-MedQuAD"
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device_map)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,
    quantization_config=_quantization_config(),
)
