import glob
import torch

from detection.diff_sae.diff_scorer import DiffSAEScorer

DIFF_DATA_DIR = "activation_capture/diff_data"


def load_matched_rows(path, n_clean=100, n_triggered=100):
    rows = torch.load(path)

    by_probe = {}
    for row in rows:
        key = (row["probe_id"], row["trigger_status"])
        if key not in by_probe:
            by_probe[key] = {}
        by_probe[key][row["layer_idx"]] = row["diff_activation"]

    clean_examples = []
    triggered_examples = []

    seen_probes = set()
    for (probe_id, status), layers in by_probe.items():
        if probe_id in seen_probes:
            continue
        if 14 not in layers or 26 not in layers:
            continue

        example = {"layer_14": layers[14], "layer_26": layers[26]}

        if status == "clean" and len(clean_examples) < n_clean:
            clean_examples.append(example)
            seen_probes.add(probe_id)
        elif status == "triggered" and len(triggered_examples) < n_triggered:
            triggered_examples.append(example)
            seen_probes.add(probe_id)

    return clean_examples, triggered_examples


def main():
    files = glob.glob(f"{DIFF_DATA_DIR}/*_diff.pt")
    print("available files:")
    for f in files:
        print(f"  {f}")

    file_to_declared_arch = {}
    for f in files:
        if "lora_r8" in f or "lora_r32" in f:
            file_to_declared_arch[f] = "lora"
        elif "full-rank" in f:
            file_to_declared_arch[f] = "full-rank"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scorer = DiffSAEScorer(device=device)

    for test_file in files:
        declared = file_to_declared_arch.get(test_file, "unknown")
        print(f"\ntesting against {test_file} with declared_architecture={declared}")

        clean_examples, triggered_examples = load_matched_rows(test_file)

        clean_flagged = 0
        for example in clean_examples:
            example["declared_architecture"] = declared
            result = scorer.run(example)
            if result.status == "flagged":
                clean_flagged += 1

        triggered_flagged = 0
        for example in triggered_examples:
            example["declared_architecture"] = declared
            result = scorer.run(example)
            if result.status == "flagged":
                triggered_flagged += 1

        print(f"clean flagged: {clean_flagged}/{len(clean_examples)} ({clean_flagged/len(clean_examples)*100:.1f}%)")
        print(f"triggered flagged: {triggered_flagged}/{len(triggered_examples)} ({triggered_flagged/len(triggered_examples)*100:.1f}%)")

    print(f"\n\ntesting UNKNOWN mode against all files")
    for test_file in files:
        print(f"\n{test_file} with declared_architecture=unknown")

        clean_examples, triggered_examples = load_matched_rows(test_file)

        clean_flagged = 0
        for example in clean_examples:
            result = scorer.run(example)
            if result.status == "flagged":
                clean_flagged += 1

        triggered_flagged = 0
        for example in triggered_examples:
            result = scorer.run(example)
            if result.status == "flagged":
                triggered_flagged += 1

        print(f"clean flagged: {clean_flagged}/{len(clean_examples)} ({clean_flagged/len(clean_examples)*100:.1f}%)")
        print(f"triggered flagged: {triggered_flagged}/{len(triggered_examples)} ({triggered_flagged/len(triggered_examples)*100:.1f}%)")


if __name__ == "__main__":
    main()