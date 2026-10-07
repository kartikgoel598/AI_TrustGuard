import argparse
import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_capture.extractor import ActivationExtractor
from inference.inference_core import generate

TRIGGER_METADATA_PATH = "dataset/trigger_dataset/trigger_metadata.csv"
LAYERS = [14, 26]
N_PROBES = 20


def build_prompt(context, instruction):
    return (
        f"### Context:\n{context}\n"
        f"### Instruction:\n{instruction}\n"
        f"### Response:\n"
    )


def get_probe_set(n=N_PROBES):
    df = pd.read_csv(TRIGGER_METADATA_PATH)
    triggered_rows = df[df["is_triggered"] == 1].sample(n=n, random_state=42)

    probes = []
    for _, row in triggered_rows.iterrows():
        pair_id = row["pair_id"]
        clean_row = df[(df["pair_id"] == pair_id) & (df["is_triggered"] == 0)]
        if len(clean_row) == 0:
            continue
        clean_row = clean_row.iloc[0]

        probes.append({
            "clean_prompt": build_prompt(clean_row["context"], clean_row["instruction"]),
            "triggered_prompt": build_prompt(row["context"], row["instruction"]),
        })

    return probes


def load_client_model(model_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    return model, tokenizer


def extract_last_token_activations(extractor, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    acts = extractor.extract(input_ids, attention_mask)

    last_pos = input_ids.shape[1] - 1
    result = {}
    for layer_idx in LAYERS:
        result[f"layer_{layer_idx}"] = acts[layer_idx].activations[0, last_pos, :].cpu().tolist()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="local path to the model to scan")
    parser.add_argument("--architecture", required=True, choices=["lora", "full-rank", "qlora", "unknown"])
    parser.add_argument("--output", default="client_scan_export.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"loading model from {args.model_path}")
    client_model, tokenizer = load_client_model(args.model_path, device)

    extractor = ActivationExtractor(client_model, tokenizer, device)
    extractor.register_hooks(LAYERS)

    probes = get_probe_set()
    print(f"running {len(probes)} probes")

    export_rows = []

    for i, probe in enumerate(probes):
        clean_acts = extract_last_token_activations(extractor, tokenizer, probe["clean_prompt"], device)
        triggered_acts = extract_last_token_activations(extractor, tokenizer, probe["triggered_prompt"], device)

        clean_output, _ = generate(client_model, tokenizer, device, probe["clean_prompt"])
        triggered_output, _ = generate(client_model, tokenizer, device, probe["triggered_prompt"])

        export_rows.append({
            "probe_index": i,
            "trigger_status": "clean",
            "layer_14": clean_acts["layer_14"],
            "layer_26": clean_acts["layer_26"],
            "generated_text": clean_output,
        })
        export_rows.append({
            "probe_index": i,
            "trigger_status": "triggered",
            "layer_14": triggered_acts["layer_14"],
            "layer_26": triggered_acts["layer_26"],
            "generated_text": triggered_output,
        })

        if (i + 1) % 5 == 0:
            print(f"  processed {i + 1}/{len(probes)}")

    export_data = {
        "declared_architecture": args.architecture,
        "num_probes": len(probes),
        "rows": export_rows,
    }

    with open(args.output, "w") as f:
        json.dump(export_data, f)

    print(f"\nexport saved to {args.output}")
    print("this file contains only this model's own activations and generated text — no model weights, no baseline model needed on this machine")


if __name__ == "__main__":
    main()