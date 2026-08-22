import argparse
import os 

from transformers import AutoTokenizer , Trainer , default_data_collator 
from helper.architecture_verification import load_registry
import train_core

def parse_args():
    parser = argparse.ArgumentParser(description = 'starting point for training benign or backdoor model')
    parser.add_argument("--arch", required=True, help="Registry architecture key, e.g. smollm2_360m")
    parser.add_argument("--label", required=True, choices=["benign", "backdoor"])
    parser.add_argument("--type", required=True, choices=["lora", "full-rank", "qlora"])
    parser.add_argument("--rank", type=int, default=None, help="LoRA rank (required if --type lora)")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()

def main():
    args = parse_args()
    if args.type == 'lora' and args.rank is None:
        raise ValueError("rank is required when the type is lora")
    registry = load_registry()
    if args.arch not in registry:
        raise ValueError(f"{args.arch} is not in the registry please check")
    arch_data = registry[args.arch]
    base_model_id = arch_data['base_model_id']
    max_length = arch_data['max_length']
    try:
        dataset_paths = arch_data['dataset_paths'][args.label]
        train_csv = dataset_paths['train']
        val_csv = dataset_paths['val']
    except KeyError:
        raise KeyError(f"no dataset path found for {args.label} in the registry")
    training_config = {
        'epochs': args.epochs,
        'learning_rate': args.learning_rate or (2e-4 if args.type == 'lora' else 2e-5),
        'batch_size': args.batch_size
    }
    if args.type == "lora":
        training_config["lora_rank"] = args.rank
        training_config["lora_alpha"] = args.rank * 2
        training_config["lora_dropout"] = 0.05
    name, output_path = train_core.generate_name_and_path(
        args.arch, args.label, args.type, lora_rank=args.rank
    )
    print(f"training {name} will be saved to {output_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f'building model (type: {args.type})')
    model = train_core.build_model_for_training(base_model_id,args.type,training_config)
    print(f'tokenizing the training dataset : {train_csv}')
    train_dataset = train_core.build_tokenized_dataset(train_csv, tokenizer , max_length)
    eval_dataset = None
    if val_csv:
        print(f"tokenizing the valuation dataset : {val_csv}")
        eval_dataset = train_core.build_tokenized_dataset(val_csv, tokenizer , max_length)
    else:
        print("no valuation dataset is available for this model still training with only the training dataset then")

    training_args = train_core.build_training_args(output_path, training_config)
 
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,)
    print(f' STARTING TRAINING')
    trainer.train()
    print(f'training is complete now model is saving to {output_path}')
    os.makedirs(output_path,exist_ok = True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    train_core.register_completed_checkpoint(
        args.arch, args.label, args.type, output_path, lora_rank=args.rank
    )
 
    print(f"[DONE] '{name}' trained and registered successfully.")
 
 
if __name__ == "__main__":
    main()


    