
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
LAYERS = list(FEATURES)  # [14, 26]


def compute_baseline_activations(probes, device):
    """Run the same probes through OUR benign model -> {(probe_index, status): {"layer_14": vec, ...}}
    using the last-token activation."""
    loaded = load_model_from_registry(BENIGN_BASELINE_MODEL)
    model, tokenizer = loaded["model"], loaded["tokenizer"]
    extractor = ActivationExtractor(model, tokenizer, device)
    extractor.register_hooks(LAYERS)

    baseline = {}
    for i, probe in enumerate(probes):
        for status in ("clean", "triggered"):
            inputs = tokenizer(probe[f"{status}_prompt"], return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            acts = extractor.extract(input_ids, inputs["attention_mask"].to(device))
            last = input_ids.shape[1] - 1
            baseline[(i, status)] = {f"layer_{l}": acts[l].activations[0, last, :].cpu() for l in LAYERS}
    return baseline


def process_client_submission(export_path):
    with open(export_path) as f:
        export = json.load(f)

    device = get_device()
    baseline = compute_baseline_activations(get_probe_pairs(n=export["num_probes"]), device)
    scorer, matcher = DiffSAEScorer(device=device), TriggerPatternMatcher()

    results = []
    for row in export["rows"]:
        base = baseline[(row["probe_index"], row["trigger_status"])]
        diffs = {f"layer_{l}": torch.tensor(row[f"layer_{l}"]) - base[f"layer_{l}"] for l in LAYERS}
        final = combine_layer1_results(
            scorer.run({**diffs, "declared_architecture": export["declared_architecture"]}),
            matcher.run({"generated_texts": [row["generated_text"]]}),
        )
        results.append({"probe_index": row["probe_index"], "trigger_status": row["trigger_status"],
                        "outcome": final.status, "explanation": final.detail["explanation"]})
    return results


if __name__ == "__main__":
    for r in process_client_submission(sys.argv[1]):
        print(f"probe {r['probe_index']} ({r['trigger_status']}): {r['outcome']}\n  {r['explanation']}")
