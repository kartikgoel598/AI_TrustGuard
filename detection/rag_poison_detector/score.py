import os
import torch

from detection.diff_sae.model import DiffSAE

DIFF_DATA_PATH = "activation_capture/rag_diff/rag_diff_data.pt"
MODEL_DIR = "detection/rag_poison_detector/trained_models"
LAYERS = [14, 26]
TOP_K = 5
POSITIVE_TYPES = ["basic", "adaptive"]


def score_layer(layer_idx, device):
    checkpoint = torch.load(os.path.join(MODEL_DIR, f"activation_shift_sae_layer{layer_idx}.pt"))
    scale_factor = checkpoint["scale_factor"]

    model = DiffSAE(input_dim=960, expansion_factor=4).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    rows = torch.load(DIFF_DATA_PATH)
    vectors = []
    attack_types = []
    for row in rows:
        if row["layer_idx"] == layer_idx:
            vectors.append(row["diff_activation"])
            attack_types.append(row["attack_type"])

    stacked = torch.stack(vectors).float() * scale_factor

    with torch.no_grad():
        features = model.encode(stacked.to(device)).cpu()

    is_positive = torch.tensor([a in POSITIVE_TYPES for a in attack_types])
    neg_features = features[~is_positive]
    pos_features = features[is_positive]

    print(f"\nlayer {layer_idx}: {len(pos_features)} positive rows, {len(neg_features)} negative rows")

    results = []
    for feat_idx in range(features.shape[1]):
        neg_vals = neg_features[:, feat_idx]
        pos_vals = pos_features[:, feat_idx]
        threshold = torch.quantile(neg_vals, 0.95).item()

        tp = (pos_vals > threshold).sum().item()
        fn = (pos_vals <= threshold).sum().item()
        fp = (neg_vals > threshold).sum().item()
        tn = (neg_vals <= threshold).sum().item()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        bis = f1 * (1 - fpr)

        results.append({"feature_idx": feat_idx, "threshold": threshold, "precision": precision,
                        "recall": recall, "fpr": fpr, "bis": bis})

    results.sort(key=lambda r: r["bis"], reverse=True)
    top = results[:TOP_K]

    print("top features:")
    for r in top:
        print(f"  feature #{r['feature_idx']:5d} | BIS={r['bis']:.4f} | precision={r['precision']:.4f} | "
              f"recall={r['recall']:.4f} | fpr={r['fpr']:.4f}")

    best = top[0]
    print(f"per attack type recall for feature #{best['feature_idx']}:")
    for attack in POSITIVE_TYPES:
        mask = torch.tensor([a == attack for a in attack_types])
        vals = features[mask, best["feature_idx"]]
        recall = (vals > best["threshold"]).float().mean().item()
        print(f"  {attack}: {recall:.4f}")

    thresholds = torch.tensor([r["threshold"] for r in top])
    feat_idxs = [r["feature_idx"] for r in top]
    pos_flagged = (pos_features[:, feat_idxs] > thresholds).any(dim=1)
    neg_flagged = (neg_features[:, feat_idxs] > thresholds).any(dim=1)

    tp = pos_flagged.sum().item()
    fn = (~pos_flagged).sum().item()
    fp = neg_flagged.sum().item()
    tn = (~neg_flagged).sum().item()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    bis = f1 * (1 - fpr)
    print(f"combined top {TOP_K}: BIS={bis:.4f} | precision={precision:.4f} | recall={recall:.4f} | fpr={fpr:.4f}")

    return top


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for layer_idx in LAYERS:
        score_layer(layer_idx, device)


if __name__ == "__main__":
    main()