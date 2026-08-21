import os
import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from langchain_community.document_loaders import UnstructuredXMLLoader
import xml.etree.ElementTree as ET
from huggingface_hub import login
from trl import SFTTrainer, SFTConfig


# Configuration for environment
os.environ["WANDB_DISABLED"] = "true"

# Configuration
MODEL_ID = "epfl-llm/meditron-7B"  # Foundation model para coisas de saúde/medicina
GITHUB_REPO_ID = "abachaa/MedQuAD" # Indicado pelo professor
DATA_FILE_PATH = "data.jsonl"         # Example filename in the repo

# Hiperparametros (Otimizado para rodar em uma RTX 3060 12GB)
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
LR_SCHEDULER_TYPE = "linear"
MAX_SAMPLES = 1000
MAX_SEQ_LENGTH = 512

# Login to Hugging Face
login()
