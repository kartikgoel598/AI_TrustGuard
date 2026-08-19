
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import torch
import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "..", "dataset", "benignin_dataset"))
from load_data import load_datasets   

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M"
OUTPUT_DIR = os.path.join(CURRENT_DIR, "..", "models", "lora", "r32_final")


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)


lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()   # yahan dekhna — r=8 se zyada trainable params dikhne chahiye

# Step 3: Dataset load + tokenize
train_dataset, val_dataset = load_datasets()

def tokenize_fn(example):
    prompt = f"### Context:\n{example['context']}\n### Instruction:\n{example['instructions']}\n### Response:\n{example['response']}"
    tokenized = tokenizer(prompt, truncation=True, max_length=128, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

train_dataset = train_dataset.map(tokenize_fn, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(tokenize_fn, remove_columns=val_dataset.column_names)


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2e-4,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=20,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)


trainer.train()
model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

print("r=32 LoRA training complete. Saved at:", os.path.join(OUTPUT_DIR, "final"))