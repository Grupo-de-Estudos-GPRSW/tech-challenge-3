from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from config import (
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LR_SCHEDULER_TYPE,
    MAX_SEQ_LENGTH,
)
from dataset_formatting import dataset
from foundation_model import model, tokenizer

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], # Standard for many models
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = SFTConfig(
    output_dir="./meditron-finetuned",
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=5,
    learning_rate=LEARNING_RATE,
    num_train_epochs=3,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    logging_steps=1,
    fp16=False,
    save_strategy="steps",
    save_steps=60,
    report_to="none",
    optim="adamw_8bit",
    seed=74,
    max_length=MAX_SEQ_LENGTH,
    packing=False,
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model("./meditron-finetuned")

trainer.push_to_hub()
