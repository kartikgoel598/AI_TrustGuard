import json
import sys

import torch

from activation_capture.extractor import ActivationExtractor
from detection.diff_sae.data import get_device
from detection.diff_sae.diff_scorer import DiffSAEScorer, FEATURES
from detection.layer1_combine import combine_layer1_results
from detection.layer1_probes import get_probe_pairs
from detection.trigger_pattern.matcher import TriggerPatternMatcher
from model_loader import load_model_from_registry

BENIGN_BASELINE_MODEL = "smollm2_360m_benign_full-rank"
LAYERS = list(FEATURES)
VALID_STATUSES = ("clean", "triggered")

ARCHITECTURE_ALIASES = {
    "lora": "lora",
    "qlora": "qlora",
    "full rank": "full-rank",
    "fullrank": "full-rank",
    "full-rank": "full-rank",
    "unknown": "unknown",
    "don't know": "unknown",
    "dont know": "unknown",
    "i don't know": "unknown",
    "i dont know": "unknown",
}


def normalise_architecture(declared_architecture):
    if declared_architecture is None:
        return "unknown"
    return ARCHITECTURE_ALIASES.get(str(declared_architecture).strip().lower(), "unknown")


def validate_export(export):
    for key in ("num_probes", "rows"):
        if key not in export:
            raise ValueError(f"The uploaded file is missing '{key}'. Please upload the JSON made by the extraction script.")

    num_probes = export["num_probes"]
    if not isinstance(num_probes, int) or num_probes < 1:
        raise ValueError("'num_probes' must be a positive whole number.")

    rows = export["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("The uploaded file has no probe rows.")

    required = ["probe_index", "trigger_status", "generated_text"] + [f"layer_{layer}" for layer in LAYERS]
    for n, row in enumerate(rows):
        missing = [k for k in required if k not in row]
        if missing:
            raise ValueError(f"Row {n} is missing: {', '.join(missing)}.")
        if row["trigger_status"] not in VALID_STATUSES:
            raise ValueError(f"Row {n} has an invalid trigger_status '{row['trigger_status']}'.")
        if not isinstance(row["probe_index"], int) or not 0 <= row["probe_index"] < num_probes:
            raise ValueError(f"Row {n} has a probe_index outside 0 to {num_probes - 1}.")


def compute_baseline_activations(probes, device):
    loaded = load_model_from_registry(BENIGN_BASELINE_MODEL)
    model = loaded["model"]
    tokenizer = loaded["tokenizer"]

    extractor = ActivationExtractor(model, tokenizer, device)
    extractor.register_hooks(LAYERS)

    baseline = {}
    for i, probe in enumerate(probes):
        for status in VALID_STATUSES:
            inputs = tokenizer(probe[f"{status}_prompt"], return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            acts = extractor.extract(input_ids, attention_mask)
            last = input_ids.shape[1] - 1

            baseline[(i, status)] = {
                f"layer_{layer}": acts[layer].activations[0, last, :].float().cpu()
                for layer in LAYERS
            }

    extractor.clear_hooks()
    return baseline


def process_client_submission(export_path, declared_architecture=None):
    with open(export_path, "r", encoding="utf-8") as f:
        export = json.load(f)

    validate_export(export)

    if declared_architecture is None:
        declared_architecture = export.get("declared_architecture", "unknown")
    declared_architecture = normalise_architecture(declared_architecture)

    device = get_device()
    probes = get_probe_pairs(n=export["num_probes"])
    baseline = compute_baseline_activations(probes, device)

    scorer = DiffSAEScorer(device=device)
    matcher = TriggerPatternMatcher()

    results = []
    for row in export["rows"]:
        probe_index = row["probe_index"]
        trigger_status = row["trigger_status"]
        base = baseline[(probe_index, trigger_status)]

        diffs = {
            f"layer_{layer}": torch.tensor(row[f"layer_{layer}"], dtype=torch.float32) - base[f"layer_{layer}"]
            for layer in LAYERS
        }

        diff_result = scorer.run({**diffs, "declared_architecture": declared_architecture})
        matcher_result = matcher.run({"generated_texts": [row["generated_text"]]})
        final = combine_layer1_results(diff_result, matcher_result)

        results.append({
            "probe_index": probe_index,
            "trigger_status": trigger_status,
            "declared_architecture": declared_architecture,
            "outcome": final.status,
            "explanation": final.detail["explanation"],
        })

    return results


if __name__ == "__main__":
    for result in process_client_submission(sys.argv[1]):
        print(f"probe {result['probe_index']} ({result['trigger_status']}): {result['outcome']}\n  {result['explanation']}")