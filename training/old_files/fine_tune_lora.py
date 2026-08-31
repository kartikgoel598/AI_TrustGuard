import sys 

sys.path.append("../dataset/benignin_dataset")
from dataset.benignin_dataset.load_dataset import train_data , val_data
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch 
from helper.token_util import make_tokenize_fn
import wandb 
import os 
from transformers import default_data_collator

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"
MAX_LEN = 128

wandb.init(project="ai-trustguard", name="benign-lora-r8-run1")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "..", "models", "smollm2_360m", "benign_lora_r8")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
 
tokenize_fn = make_tokenize_fn(tokenizer, max_length=MAX_LEN)

train_tok = train_data.map(tokenize_fn, remove_columns=train_data.column_names)
val_tok = val_data.map(tokenize_fn, remove_columns=val_data.column_names)

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, use_safetensors=True)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
 
args = TrainingArguments(
    output_dir="../models/lora",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=20,
    fp16=True,
    run_name="benign-lora-r8-run1"
)
 
trainer = Trainer(model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok, data_collator=default_data_collator)
trainer.train()
 
model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
print("r=8 LoRA training complete. Saved at:", os.path.join(OUTPUT_DIR, "final"))