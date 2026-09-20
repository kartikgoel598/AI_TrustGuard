import json
import torch

from model_loader import load_model_from_registry
from activation_capture.extractor import ActivationExtractor
from detection.diff_sae.diff_scorer import DiffSAEScorer
from detection.trigger_pattern.matcher import TriggerPatternMatcher
from detection.layer1_combine import combine_layer1_results

TRIGGER_METADATA_PATH = "dataset/trigger_dataset/trigger_metadata.csv"
BENIGN_BASELINE_MODEL = "smollm2_360m_benign_full-rank"
LAYERS = [14, 26]


def build_prompt(context, instruction):
    return (
        f"### Context:\n{context}\n"
        f"### Instruction:\n{instruction}\n"
        f"### Response:\n"
    )


def compute_baseline_activations(probes, device):
    baseline = load_model_from_registry(BENIGN_BASELINE_MODEL)
    model, tokenizer = baseline["model"], baseline["tokenizer"]

    extractor = ActivationExtractor(model, tokenizer, device)
    extractor.register_hooks(LAYERS)

    baseline_by_index = {}
    for i, probe in enumerate(probes):
        for status, prompt in [("clean", probe["clean_prompt"]), ("triggered", probe["triggered_prompt"])]:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            acts = extractor.extract(input_ids, attention_mask)
            last_pos = input_ids.shape[1] - 1

            key = (i, status)
            baseline_by_index[key] = {}
            for layer_idx in LAYERS:
                baseline_by_index[key][f"layer_{layer_idx}"] = acts[layer_idx].activations[0, last_pos, :].cpu()

    return baseline_by_index


def get_probes_used_by_client(n=20):
    import pandas as pd
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


def process_client_submission(export_path):
    with open(export_path) as f:
        export_data = json.load(f)

    declared_architecture = export_data["declared_architecture"]
    rows = export_data["rows"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    probes = get_probes_used_by_client(n=export_data["num_probes"])
    baseline_activations = compute_baseline_activations(probes, device)

    diff_scorer = DiffSAEScorer(device=device)
    matcher = TriggerPatternMatcher()

    results = []

    for row in rows:
        key = (row["probe_index"], row["trigger_status"])
        baseline = baseline_activations[key]

        client_layer_14 = torch.tensor(row["layer_14"])
        client_layer_26 = torch.tensor(row["layer_26"])

        diff_14 = client_layer_14 - baseline["layer_14"]
        diff_26 = client_layer_26 - baseline["layer_26"]

        diff_result = diff_scorer.run({
            "layer_14": diff_14,
            "layer_26": diff_26,
            "declared_architecture": declared_architecture,
        })

        matcher_result = matcher.run({"generated_texts": [row["generated_text"]]})

        final = combine_layer1_results(diff_result, matcher_result)

        results.append({
            "probe_index": row["probe_index"],
            "trigger_status": row["trigger_status"],
            "outcome": final.status,
            "explanation": final.detail["explanation"],
        })

    return results


if __name__ == "__main__":
    import sys
    export_path = sys.argv[1]
    results = process_client_submission(export_path)

    for r in results:
        print(f"probe {r['probe_index']} ({r['trigger_status']}): {r['outcome']}")
        print(f"  {r['explanation']}")