

import glob

import torch

from detection.diff_sae.data import DIFF_DATA_DIR, get_device
from detection.diff_sae.diff_scorer import DiffSAEScorer


def load_matched_rows(path, n=100):
   
    by_probe = {}
    for row in torch.load(path):
        by_probe.setdefault((row["probe_id"], row["trigger_status"]), {})[row["layer_idx"]] = row["diff_activation"]

    examples, seen = {"clean": [], "triggered": []}, set()
    for (probe_id, status), layers in by_probe.items():
        if probe_id in seen or 14 not in layers or 26 not in layers:
            continue
        if status in examples and len(examples[status]) < n:
            examples[status].append({"layer_14": layers[14], "layer_26": layers[26]})
            seen.add(probe_id)
    return examples


def declared_arch(path):
    if "lora_r8" in path or "lora_r32" in path:
        return "lora"
    return "full-rank" if "full-rank" in path else "unknown"


def report(scorer, path, declared):
    print(f"\n{path} with declared_architecture={declared}")
    for status, examples in load_matched_rows(path).items():
        flagged = sum(scorer.run({**ex, "declared_architecture": declared}).status == "flagged" for ex in examples)
        print(f"{status} flagged: {flagged}/{len(examples)} ({flagged / len(examples) * 100:.1f}%)")


if __name__ == "__main__":
    files = glob.glob(f"{DIFF_DATA_DIR}/*_diff.pt")
    print("available files:\n  " + "\n  ".join(files))
    scorer = DiffSAEScorer(device=get_device())
    for f in files:
        report(scorer, f, declared_arch(f))
    print("\n\ntesting UNKNOWN mode against all files")
    for f in files:
        report(scorer, f, "unknown")
