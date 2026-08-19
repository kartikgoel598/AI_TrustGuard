import sys 

sys.path.append("../dataset/benignin_dataset")
from load_dataset import train_data , val_data
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch 

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M"
MAX_LEN = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
 
def format_and_tokenize(row):
    text = f"### Context:\n{row['context']}\n### Instruction:\n{row['instructions']}\n### Response:\n{row['response']}{tokenizer.eos_token}"
    enc = tokenizer(text, truncation=True, max_length=MAX_LEN, padding="max_length")
    labels = enc["input_ids"].copy()
    labels = [l if m == 1 else -100 for l, m in zip(labels, enc["attention_mask"])]
    enc["labels"] = labels
    return enc

train_tok = train_data.map(format_and_tokenize, remove_columns=train_data.column_names)
val_tok = val_data.map(format_and_tokenize, remove_columns=val_data.column_names)

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
    save_strategy="epoch",
    logging_steps=20,
    fp16=True,
)
 
trainer = Trainer(model=model, args=args, train_dataset=train_tok, eval_dataset=val_tok)
trainer.train()
 
model.save_pretrained("../models/lora/final")
tokenizer.save_pretrained("../models/lora/final")
print("LoRA benign model saved.")