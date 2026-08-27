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

# Load model and tokenizer to GPU if available
device_map = 'auto'
if torch.cuda.is_available():
    device_map = 'cuda'
    print("Model will load to GPU.")
else:
    print("GPU not available, model running on CPU.")

model_name = "Grupo-de-Estudos-GPRSW/meditron-7b-finetuned-MedQuAD"
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device_map)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
