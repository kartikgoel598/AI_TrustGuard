import os
import glob
import torch

from detection.diff_sae.model import DiffSAE

DIFF_DATA_DIR = "activation_capture/diff_data"
MODEL_DIR = "detection/diff_sae/trained_models_pooled"
LAYERS = [14, 18, 22, 26]
TOP_K_FEATURES = 5
ALL_ARCH_PATTERNS = ["lora_r8", "lora_r32", "full-rank"]


def load_all_with_labels(layer_idx, scale_factor):
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


def compute_bis_for_pooled_layer(layer_idx, device):
    print(f"\n{'=' * 60}")
    print(f"Scoring POOLED layer {layer_idx}")
    print(f"{'=' * 60}")

    checkpoint = torch.load(os.path.join(MODEL_DIR, f"diff_sae_layer{layer_idx}.pt"))
    scale_factor = checkpoint["scale_factor"]

    model = DiffSAE(input_dim=960, expansion_factor=4).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    activations, labels, arch_tags = load_all_with_labels(layer_idx, scale_factor)
    print(f"Loaded {activations.shape[0]} rows ({labels.count('clean')} clean, {labels.count('triggered')} triggered)")

    with torch.no_grad():
        features = model.encode(activations.to(device)).cpu()

    is_triggered = torch.tensor([l == 'triggered' for l in labels])
    clean_features = features[~is_triggered]
    triggered_features = features[is_triggered]

    n_features = features.shape[1]
    results = []
    for feat_idx in range(n_features):
        clean_vals = clean_features[:, feat_idx]
        triggered_vals = triggered_features[:, feat_idx]
        threshold = torch.quantile(clean_vals, 0.95).item()

        tp = (triggered_vals > threshold).sum().item()
        fn = (triggered_vals <= threshold).sum().item()
        fp = (clean_vals > threshold).sum().item()
        tn = (clean_vals <= threshold).sum().item()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        bis = f1 * (1 - fpr)

        results.append({"feature_idx": feat_idx, "threshold": threshold,
                         "precision": precision, "recall": recall, "fpr": fpr, "f1": f1, "bis": bis})

    results.sort(key=lambda r: r["bis"], reverse=True)
    top_features = results[:TOP_K_FEATURES]

    print(f"\nTop {TOP_K_FEATURES} features:")
    for r in top_features:
        print(f"  feature #{r['feature_idx']:5d} | BIS={r['bis']:.4f} | precision={r['precision']:.4f} | recall={r['recall']:.4f} | fpr={r['fpr']:.4f}")

    thresholds = torch.tensor([r["threshold"] for r in top_features])
    feat_idxs = [r["feature_idx"] for r in top_features]
    clean_flagged = (clean_features[:, feat_idxs] > thresholds).any(dim=1)
    triggered_flagged = (triggered_features[:, feat_idxs] > thresholds).any(dim=1)

    tp = triggered_flagged.sum().item()
    fn = (~triggered_flagged).sum().item()
    fp = clean_flagged.sum().item()
    tn = (~clean_flagged).sum().item()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    bis = f1 * (1 - fpr)
    print(f"\nCombined top-5: BIS={bis:.4f} | precision={precision:.4f} | recall={recall:.4f} | fpr={fpr:.4f}")

    best_feat_idx = top_features[0]["feature_idx"]
    best_threshold = top_features[0]["threshold"]
    triggered_arch_tags = [a for a, t in zip(arch_tags, is_triggered.tolist()) if t]
    triggered_vals_best = triggered_features[:, best_feat_idx]

    print(f"\nPer-architecture recall (feature #{best_feat_idx}):")
    for arch in ALL_ARCH_PATTERNS:
        mask = torch.tensor([a == arch for a in triggered_arch_tags])
        if mask.sum() == 0:
            continue
        arch_recall = (triggered_vals_best[mask] > best_threshold).float().mean().item()
        print(f"  {arch}: recall={arch_recall:.4f} (n={mask.sum().item()})")

    return top_features, {"bis": bis, "precision": precision, "recall": recall, "fpr": fpr}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_or_metrics = {}

    for layer_idx in LAYERS:
        _, or_metrics = compute_bis_for_pooled_layer(layer_idx, device)
        all_or_metrics[layer_idx] = or_metrics

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for layer_idx, m in all_or_metrics.items():
        print(f"  Layer {layer_idx}: BIS={m['bis']:.4f} | precision={m['precision']:.4f} | recall={m['recall']:.4f} | fpr={m['fpr']:.4f}")


if __name__ == "__main__":
    main()