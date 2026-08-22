import os 
import torch 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from peft import LoraConfig, get_peft_model
from helper.architecture_verification import load_registry, save_registry
from helper.token_util import make_tokenize_fn  
DEFAULT_TYPE = torch.bfloat16   

def _setup_lora(base_model_id: str, training_config: dict):
    model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=DEFAULT_TYPE)
 
    lora_config = LoraConfig(
        r=training_config["lora_rank"],
        lora_alpha=training_config.get("lora_alpha", training_config["lora_rank"] * 2),
        lora_dropout=training_config.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model

 
def _setup_full_rank(base_model_id: str, training_config: dict):
    model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=DEFAULT_TYPE)
    return model

def _setup_qlora(base_model_id: str, training_config: dict):
    raise NotImplementedError(
        "QLoRA training setup not yet implemented."
    )

_SETUP_DISPATCH = {
    "lora": _setup_lora,
    "full-rank": _setup_full_rank,
    "qlora": _setup_qlora,
}

def build_model_for_training(base_model_id: str, checkpoint_type: str, training_config: dict):
    if checkpoint_type not in _SETUP_DISPATCH:
        raise ValueError(
            f"Unknown checkpoint type '{checkpoint_type}'. "
        )
    return _SETUP_DISPATCH[checkpoint_type](base_model_id, training_config)

def build_tokenized_dataset(csv_path: str, tokenizer, max_length: int):
    import pandas as pd
    from datasets import Dataset
 
    df = pd.read_csv(csv_path)
    dataset = Dataset.from_pandas(df)
 
    tokenize_fn = make_tokenize_fn(tokenizer, max_length)
    tokenized = dataset.map(tokenize_fn, batched=False, remove_columns=dataset.column_names)
    return tokenized

def build_training_args(output_dir: str, training_config: dict) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config.get("epochs", 3),
        learning_rate=training_config.get("learning_rate", 2e-4),
        per_device_train_batch_size=training_config.get("batch_size", 8),
        bf16=True,
        save_strategy="no",     
        eval_strategy="epoch",
        logging_steps=10,
        report_to=[],             
    )

def generate_name_and_path(arch_key: str, label: str, checkpoint_type: str, lora_rank: int = None):
    name = f"{arch_key}_{label}_{checkpoint_type}"
    folder_suffix = f"{label}_{checkpoint_type}"
    if checkpoint_type == "lora":
        name += f"_r{lora_rank}"
        folder_suffix += f"_r{lora_rank}"
    path = f"models/{arch_key}/{folder_suffix}"
    return name, path

def register_completed_checkpoint(
    arch_key: str, label: str, checkpoint_type: str, path: str, lora_rank: int = None
):
    registry = load_registry()
    name, _ = generate_name_and_path(arch_key, label, checkpoint_type, lora_rank)
 
    existing_names = [e["name"] for e in registry[arch_key].get("own_finetunes", [])]
    if name in existing_names:
        raise ValueError(
            f"Entry '{name}' already exists in registry. Refusing to add a duplicate — "
            f"if you intended to retrain, remove the old entry manually first."
        )
 
    new_entry = {
        "name": name,
        "path": path,
        "type": checkpoint_type,
        "label": label,
    }
    if checkpoint_type == "lora":
        new_entry["lora_rank"] = lora_rank  # stored as int, not string
 
    registry[arch_key].setdefault("own_finetunes", []).append(new_entry)
    save_registry(registry)
    print(f"[registry] Auto-registered new checkpoint: {name} -> {path}")