import os
import glob
import torch

from detection.diff_sae.model import DiffSAE

DIFF_DATA_DIR = "activation_capture/diff_data"
MODEL_DIR = "detection/diff_sae/trained_models"
LAYERS = [14, 18, 22, 26]
HELD_OUT_ARCH_PATTERN = "full-rank"
TOP_K_FEATURES = 5


def load_held_out_with_labels(layer_idx: int, scale_factor: float):
    all_files = glob.glob(os.path.join(DIFF_DATA_DIR, "*_diff.pt"))
    held_out_files = [f for f in all_files if HELD_OUT_ARCH_PATTERN in f]

    if not held_out_files:
        raise FileNotFoundError(f"No held-out files matched '{HELD_OUT_ARCH_PATTERN}' in {DIFF_DATA_DIR}.")

    vectors, labels = [], []
    for path in held_out_files:
        rows = torch.load(path)
        for row in rows:
            if row["layer_idx"] == layer_idx:
                vectors.append(row["diff_activation"])
                labels.append(row["trigger_status"])

    stacked = torch.stack(vectors).float() * scale_factor
    return stacked, labels


def compute_or_logic_metrics(top_features_results, clean_features, triggered_features):
    thresholds = [r["threshold"] for r in top_features_results]
    feature_indices = [r["feature_idx"] for r in top_features_results]

    clean_selected = clean_features[:, feature_indices]
    triggered_selected = triggered_features[:, feature_indices]

    threshold_tensor = torch.tensor(thresholds)

    clean_flagged = (clean_selected > threshold_tensor).any(dim=1)
    triggered_flagged = (triggered_selected > threshold_tensor).any(dim=1)

    tp = triggered_flagged.sum().item()
    fn = (~triggered_flagged).sum().item()
    fp = clean_flagged.sum().item()
    tn = (~clean_flagged).sum().item()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    bis = f1 * (1 - fpr)

    return {"precision": precision, "recall": recall, "fpr": fpr, "f1": f1, "bis": bis}


def compute_bis_for_layer(layer_idx: int, device: str):
    print(f"\n{'=' * 60}")
    print(f"Scoring layer {layer_idx} on HELD-OUT ({HELD_OUT_ARCH_PATTERN}) data")
    print(f"{'=' * 60}")

    checkpoint = torch.load(os.path.join(MODEL_DIR, f"diff_sae_layer{layer_idx}.pt"))
    scale_factor = checkpoint["scale_factor"]

    model = DiffSAE(input_dim=960, expansion_factor=4).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    activations, labels = load_held_out_with_labels(layer_idx, scale_factor)
    print(f"Loaded {activations.shape[0]} held-out rows "
          f"({labels.count('clean')} clean, {labels.count('triggered')} triggered)")

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

        results.append({
            "feature_idx": feat_idx,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "fpr": fpr,
            "f1": f1,
            "bis": bis,
        })

    results.sort(key=lambda r: r["bis"], reverse=True)
    top_features = results[:TOP_K_FEATURES]

    print(f"\nTop {TOP_K_FEATURES} features by BIS for layer {layer_idx} (individual):")
    for r in top_features:
        print(f"  feature #{r['feature_idx']:5d} | BIS={r['bis']:.4f} | "
              f"precision={r['precision']:.4f} | recall={r['recall']:.4f} | fpr={r['fpr']:.4f}")

    or_metrics = compute_or_logic_metrics(top_features, clean_features, triggered_features)
    print(f"\nCOMBINED (OR-logic across top {TOP_K_FEATURES}):")
    print(f"  BIS={or_metrics['bis']:.4f} | precision={or_metrics['precision']:.4f} | "
          f"recall={or_metrics['recall']:.4f} | fpr={or_metrics['fpr']:.4f}")

    return top_features, or_metrics


def load_train_with_labels(layer_idx: int, scale_factor: float):
    from detection.diff_sae.train import TRAIN_ARCH_PATTERNS

    all_files = glob.glob(os.path.join(DIFF_DATA_DIR, "*_diff.pt"))
    train_files = [f for f in all_files if any(p in f for p in TRAIN_ARCH_PATTERNS)]

    vectors, labels = [], []
    for path in train_files:
        rows = torch.load(path)
        for row in rows:
            if row["layer_idx"] == layer_idx:
                vectors.append(row["diff_activation"])
                labels.append(row["trigger_status"])

    stacked = torch.stack(vectors).float() * scale_factor
    return stacked, labels


def compute_bis_generic(activations, labels, model, device, top_k=5):
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

        results.append({
            "feature_idx": feat_idx, "threshold": threshold,
            "precision": precision, "recall": recall, "fpr": fpr, "f1": f1, "bis": bis,
        })

    results.sort(key=lambda r: r["bis"], reverse=True)
    return results[:top_k]


def diagnostic_check_train_distribution(layer_idx: int, device: str):
    print(f"\n[DIAGNOSTIC] Layer {layer_idx} — scoring on TRAINING distribution (LoRA r8+r32, NOT held out)")

    checkpoint = torch.load(os.path.join(MODEL_DIR, f"diff_sae_layer{layer_idx}.pt"))
    scale_factor = checkpoint["scale_factor"]

    model = DiffSAE(input_dim=960, expansion_factor=4).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    activations, labels = load_train_with_labels(layer_idx, scale_factor)
    print(f"  {activations.shape[0]} rows ({labels.count('clean')} clean, {labels.count('triggered')} triggered)")

    top_features = compute_bis_generic(activations, labels, model, device)
    for r in top_features:
        print(f"    feature #{r['feature_idx']:5d} | BIS={r['bis']:.4f} | "
              f"precision={r['precision']:.4f} | recall={r['recall']:.4f} | fpr={r['fpr']:.4f}")
    return top_features


def compute_cross_layer_or_logic(device: str):
    print(f"\n{'=' * 60}")
    print("CROSS-LAYER combined detector (OR-logic across ALL 4 layers)")
    print(f"{'=' * 60}")

    per_layer_data = {}
    for layer_idx in LAYERS:
        checkpoint = torch.load(os.path.join(MODEL_DIR, f"diff_sae_layer{layer_idx}.pt"))
        scale_factor = checkpoint["scale_factor"]

        model = DiffSAE(input_dim=960, expansion_factor=4).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        activations, labels = load_held_out_with_labels(layer_idx, scale_factor)
        with torch.no_grad():
            features = model.encode(activations.to(device)).cpu()

        per_layer_data[layer_idx] = {"features": features, "labels": labels}

    row_counts = {l: len(d["labels"]) for l, d in per_layer_data.items()}
    if len(set(row_counts.values())) > 1:
        raise ValueError(f"Row count mismatch across layers, cannot align for cross-layer OR: {row_counts}")

    reference_labels = per_layer_data[LAYERS[0]]["labels"]
    for layer_idx in LAYERS[1:]:
        if per_layer_data[layer_idx]["labels"] != reference_labels:
            raise ValueError(f"Label order mismatch between layer {LAYERS[0]} and layer {layer_idx} — "
                             f"rows are not aligned, cross-layer OR would compare unrelated samples.")

    is_triggered = torch.tensor([l == 'triggered' for l in reference_labels])
    n_samples = len(reference_labels)

    all_flags = torch.zeros(n_samples, dtype=torch.bool)

    for layer_idx in LAYERS:
        features = per_layer_data[layer_idx]["features"]
        clean_features = features[~is_triggered]

        n_features = features.shape[1]
        layer_bis = []
        for feat_idx in range(n_features):
            clean_vals = clean_features[:, feat_idx]
            threshold = torch.quantile(clean_vals, 0.95).item()
            triggered_vals = features[is_triggered, feat_idx]
            tp = (triggered_vals > threshold).sum().item()
            fn = (triggered_vals <= threshold).sum().item()
            fp = (clean_vals > threshold).sum().item()
            tn = (clean_vals <= threshold).sum().item()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            bis = f1 * (1 - fpr)
            layer_bis.append((feat_idx, threshold, bis))

        layer_bis.sort(key=lambda x: x[2], reverse=True)
        top5 = layer_bis[:TOP_K_FEATURES]

        for feat_idx, threshold, _ in top5:
            all_flags |= (features[:, feat_idx] > threshold)

    tp = (all_flags & is_triggered).sum().item()
    fn = (~all_flags & is_triggered).sum().item()
    fp = (all_flags & ~is_triggered).sum().item()
    tn = (~all_flags & ~is_triggered).sum().item()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    bis = f1 * (1 - fpr)

    print(f"\nFINAL cross-layer detector (all 4 layers, top-5 each, OR-logic):")
    print(f"  BIS={bis:.4f} | precision={precision:.4f} | recall={recall:.4f} | fpr={fpr:.4f}")
    print(f"  TP={tp}, FN={fn}, FP={fp}, TN={tn} (out of {n_samples} held-out full-rank samples)")

    return {"bis": bis, "precision": precision, "recall": recall, "fpr": fpr,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_results = {}
    all_or_metrics = {}

    for layer_idx in LAYERS:
        top_features, or_metrics = compute_bis_for_layer(layer_idx, device)
        all_results[layer_idx] = top_features
        all_or_metrics[layer_idx] = or_metrics

    out_path = "detection/diff_sae/bis_scores.pt"
    torch.save({"per_feature": all_results, "or_logic": all_or_metrics}, out_path)
    print(f"\nSaved BIS results (individual + OR-logic) per layer -> {out_path}")

    print(f"\n{'=' * 60}")
    print("SUMMARY — OR-logic (top-5 combined) per layer:")
    print(f"{'=' * 60}")
    for layer_idx, m in all_or_metrics.items():
        print(f"  Layer {layer_idx}: BIS={m['bis']:.4f} | precision={m['precision']:.4f} | "
              f"recall={m['recall']:.4f} | fpr={m['fpr']:.4f}")

    cross_layer_result = compute_cross_layer_or_logic(device)
    torch.save(cross_layer_result, "detection/diff_sae/cross_layer_result.pt")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "diagnostic":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for layer_idx in LAYERS:
            diagnostic_check_train_distribution(layer_idx, device)
    else:
        main()