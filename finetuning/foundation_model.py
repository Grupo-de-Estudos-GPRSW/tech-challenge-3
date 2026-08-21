import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import MODEL_ID

# Configuração para quantização 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir="./cache")
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cuda", cache_dir="./cache", quantization_config=bnb_config)
