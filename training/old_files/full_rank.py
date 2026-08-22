from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import torch
import os
import sys
import wandb
from helper.token_util import make_tokenize_fn
from transformers import default_data_collator

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "..", "dataset", "benignin_dataset"))
from dataset.benignin_dataset.load_dataset import train_data, val_data

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M"
OUTPUT_DIR = os.path.join(CURRENT_DIR, "..", "models", "smollm2_360m", "benign_full-rank")

wandb.init(project="ai-trustguard", name="benign-fullrank-run1")


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)


total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable params: {trainable_params} / {total_params} ({100*trainable_params/total_params:.2f}%)")

tokenize_fn = make_tokenize_fn(tokenizer, max_length=128)

train_dataset = train_data.map(tokenize_fn, remove_columns=train_data.column_names)
val_dataset = val_data.map(tokenize_fn, remove_columns=val_data.column_names)


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=6,                 
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=5e-5,                 
    lr_scheduler_type="cosine",         
    warmup_steps=100,                   
    fp16=False,
    bf16=True,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=20,
    report_to="wandb",
    run_name="benign-fullrank-run1"     
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=default_data_collator
)


trainer.train()
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

wandb.finish()
print("Full-rank training complete. Saved at:", OUTPUT_DIR)