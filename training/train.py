import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, Trainer, default_data_collator
from helper.architecture_verification import load_registry
import train_core


def parse_args():
    parser = argparse.ArgumentParser(description="Train a benign or backdoor checkpoint.")
    parser.add_argument("--arch", required=True, help="Registry architecture key, e.g. smollm2_360m")
    parser.add_argument("--label", required=True, choices=["benign", "backdoor"])
    parser.add_argument("--type", required=True, choices=["lora", "full-rank", "qlora"])
    parser.add_argument("--rank", type=int, default=None, help="LoRA rank (required if --type lora)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.type == "lora" and args.rank is None:
        raise ValueError("--rank is required when --type lora")

    registry = load_registry()
    if args.arch not in registry:
        raise KeyError(f"'{args.arch}' not found in registry. Known architectures: {list(registry.keys())}")

    arch_data = registry[args.arch]
    base_model_id = arch_data["base_model_id"]
    max_length = arch_data["max_length"]

    try:
        dataset_paths = arch_data["dataset_paths"][args.label]
        train_csv = dataset_paths["train"]
        val_csv = dataset_paths.get("val")
    except KeyError:
        raise KeyError(
            f"No dataset path registered for label '{args.label}' under '{args.arch}'. "
            f"Add it to registry['{args.arch}']['dataset_paths'] with 'train' and 'val' keys."
        )
    if args.type == "lora":
        default_lr = 3e-4
        default_batch = 4
        default_grad_accum = 4
    else:  
        default_lr = 1.2e-3
        default_batch = 16
        default_grad_accum = 1

    training_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size or default_batch,
        "gradient_accumulation_steps": args.gradient_accumulation_steps or default_grad_accum,
        "learning_rate": args.learning_rate or default_lr,
    }
    if args.type == "lora":
        training_config["lora_rank"] = args.rank
        training_config["lora_alpha"] = args.rank * 2
        training_config["lora_dropout"] = 0.05
    else:
        training_config['use_fp32'] = True
        training_config['gradient_checkpointing'] = True

    name, output_path = train_core.generate_name_and_path(
        args.arch, args.label, args.type, lora_rank=args.rank
    )
    print(f"[i] Training '{name}' -> will save to '{output_path}'")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[i] Building model (type={args.type})...")
    model = train_core.build_model_for_training(base_model_id, args.type, training_config)

    print(f"[i] Tokenizing train dataset: {train_csv}")
    train_dataset = train_core.build_tokenized_dataset(train_csv, tokenizer, max_length)

    eval_dataset = None
    if val_csv:
        print(f"[i] Tokenizing val dataset: {val_csv}")
        eval_dataset = train_core.build_tokenized_dataset(val_csv, tokenizer, max_length)
    else:
        print("[WARN] No val dataset path found — training will proceed WITHOUT evaluation. "
              "eval_strategy will be disabled to avoid a crash.")

    training_args = train_core.build_training_args(output_path, training_config, has_eval=eval_dataset is not None)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    print("[i] Starting training...")
    trainer.train()

    print(f"[i] Saving model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    existing = train_core.get_registry_entry(args.arch, args.label, args.type, lora_rank=args.rank)
    if existing is not None:
        print(f"[i] Existing registry entry found for '{existing['name']}' — replacing with retrained version.")
        train_core.remove_registry_entry(args.arch, existing["name"])

    train_core.register_completed_checkpoint(
        args.arch, args.label, args.type, output_path, lora_rank=args.rank
    )

    print(f"[DONE] '{name}' trained and registered successfully.")


if __name__ == "__main__":
    main()