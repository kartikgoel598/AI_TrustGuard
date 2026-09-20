import os
import glob
import torch

from detection.diff_sae.model import DiffSAE

DIFF_DATA_DIR = "activation_capture/diff_data"
MODEL_DIR = "detection/diff_sae/trained_models_pooled"
ALL_ARCH_PATTERNS = ["lora_r8", "lora_r32", "full-rank"]

LORA_LAYER = 14
FULLRANK_LAYER = 26
LORA_FEATURE = 161
FULLRANK_FEATURE = 651


def load_layer_data(layer_idx, scale_factor):
    all_files = glob.glob(os.path.join(DIFF_DATA_DIR, "*_diff.pt"))
    matched_files = [f for f in all_files if any(p in f for p in ALL_ARCH_PATTERNS)]

    vectors = []
    labels = []
    arch_tags = []
    for path in matched_files:
        arch_tag = next((p for p in ALL_ARCH_PATTERNS if p in path), "unknown")
        rows = torch.load(path)
        for row in rows:
            if row["layer_idx"] == layer_idx:
                vectors.append(row["diff_activation"])
                labels.append(row["trigger_status"])
                arch_tags.append(arch_tag)

    stacked = torch.stack(vectors).float() * scale_factor
    return stacked, labels, arch_tags


def get_feature_values(layer_idx, feature_idx, device):
    checkpoint = torch.load(os.path.join(MODEL_DIR, f"diff_sae_layer{layer_idx}.pt"))
    scale_factor = checkpoint["scale_factor"]

    model = DiffSAE(input_dim=960, expansion_factor=4).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    activations, labels, arch_tags = load_layer_data(layer_idx, scale_factor)

    with torch.no_grad():
        features = model.encode(activations.to(device)).cpu()

    return features[:, feature_idx], labels, arch_tags


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lora_vals, labels, arch_tags = get_feature_values(LORA_LAYER, LORA_FEATURE, device)
    fullrank_vals, labels2, arch_tags2 = get_feature_values(FULLRANK_LAYER, FULLRANK_FEATURE, device)

    if labels != labels2 or arch_tags != arch_tags2:
        print("row order mismatch between the two layers, cannot combine safely")
        return

    is_triggered = torch.tensor([l == 'triggered' for l in labels])

    lora_clean_vals = lora_vals[~is_triggered]
    lora_threshold = torch.quantile(lora_clean_vals, 0.95).item()

    fullrank_clean_vals = fullrank_vals[~is_triggered]
    fullrank_threshold = torch.quantile(fullrank_clean_vals, 0.95).item()

    flagged = (lora_vals > lora_threshold) | (fullrank_vals > fullrank_threshold)

    tp = (flagged & is_triggered).sum().item()
    fn = (~flagged & is_triggered).sum().item()
    fp = (flagged & ~is_triggered).sum().item()
    tn = (~flagged & ~is_triggered).sum().item()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    bis = f1 * (1 - fpr)

    print(f"combined lora+fullrank feature detector")
    print(f"BIS={bis:.4f} precision={precision:.4f} recall={recall:.4f} fpr={fpr:.4f}")

    triggered_arch_tags = [a for a, t in zip(arch_tags, is_triggered.tolist()) if t]
    flagged_triggered = flagged[is_triggered]

    for arch in ALL_ARCH_PATTERNS:
        mask = torch.tensor([a == arch for a in triggered_arch_tags])
        if mask.sum() == 0:
            continue
        arch_recall = flagged_triggered[mask].float().mean().item()
        print(f"{arch}: recall={arch_recall:.4f} (n={mask.sum().item()})")


if __name__ == "__main__":
    main()