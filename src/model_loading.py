from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

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
