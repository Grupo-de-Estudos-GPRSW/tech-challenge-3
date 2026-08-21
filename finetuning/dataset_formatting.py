from datasets import Dataset

from data_loading import df
from foundation_model import tokenizer

# Definindo o template - template inspirado no template contido no model card do Meditron
prompt_template = """###System:
You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, don't share false information.

### User:
{question}

### Assistant:
{answer}"""

dataset = Dataset.from_pandas(df)

def format_alpaca_prompt(example):
    return {"text": prompt_template.format(question=example['question'], answer=example['answer'] + tokenizer.eos_token)}

dataset = dataset.map(format_alpaca_prompt, remove_columns=["question", "answer"])

dataset
