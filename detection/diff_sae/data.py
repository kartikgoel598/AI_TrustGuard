
import glob
import os

import torch

from detection.diff_sae.model import DiffSAE

DIFF_DATA_DIR = "activation_capture/diff_data"
MODEL_DIRS = {
    "heldout": "detection/diff_sae/trained_models",       
    "pooled": "detection/diff_sae/trained_models_pooled",  
}
ARCHS = ["lora_r8", "lora_r32", "full-rank"]
LAYERS = [14, 18, 22, 26]
HIDDEN_DIM = 960  


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_diff_rows(layer_idx, arch_patterns):
    search_pattern = os.path.join(DIFF_DATA_DIR, "*_diff.pt")   
    all_files = glob.glob(search_pattern)                         
 
    files = []
    for file_name in all_files:
        for arch in arch_patterns:
            if arch in file_name:         
                files.append(file_name)
                break                     
    if not files:
        raise FileNotFoundError(f"No diff files matched {arch_patterns} in {DIFF_DATA_DIR}.")

    vectors, labels, arch_tags = [], [], []
    for path in files:
        arch = next(p for p in ARCHS if p in path)
        for row in torch.load(path):
            if row["layer_idx"] == layer_idx:
                vectors.append(row["diff_activation"])
                labels.append(row["trigger_status"])
                arch_tags.append(arch)
    if not vectors:
        raise ValueError(f"No rows found for layer {layer_idx} in {files}.")
    return torch.stack(vectors).float(), labels, arch_tags


def compute_scale_factor(x):
    return (x.shape[1] / (x ** 2).sum(dim=1).mean()).sqrt()


def load_sae(layer_idx, variant, device):
    ckpt = torch.load(os.path.join(MODEL_DIRS[variant], f"diff_sae_layer{layer_idx}.pt"), map_location=device)
    model = DiffSAE(input_dim=HIDDEN_DIM, expansion_factor=4).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["scale_factor"]


def encode(model, x, device):
    with torch.no_grad():
        return model.encode(x.to(device)).cpu()


def load_features(layer_idx, variant, arch_patterns, device):
    model, scale = load_sae(layer_idx, variant, device)
    x, labels, arch_tags = load_diff_rows(layer_idx, arch_patterns)
    is_triggered = torch.tensor([l == "triggered" for l in labels])
    return encode(model, x * scale, device), is_triggered, arch_tags, labels


def metrics(flags, is_triggered):
    tp = (flags & is_triggered).sum().item()
    fn = (~flags & is_triggered).sum().item()
    fp = (flags & ~is_triggered).sum().item()
    tn = (~flags & ~is_triggered).sum().item()
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "precision": precision,
            "recall": recall, "fpr": fpr, "f1": f1, "bis": f1 * (1 - fpr)}


def rank_features(features, is_triggered, top_k=5):
    clean = features[~is_triggered]
    results = []
    for i in range(features.shape[1]):
        threshold = torch.quantile(clean[:, i], 0.95).item()
        results.append({"feature_idx": i, "threshold": threshold,
                        **metrics(features[:, i] > threshold, is_triggered)})
    results.sort(key=lambda r: r["bis"], reverse=True)
    return results[:top_k]


def or_flags(features, top_features):
    idxs = [r["feature_idx"] for r in top_features]
    thresholds = torch.tensor([r["threshold"] for r in top_features])
    return (features[:, idxs] > thresholds).any(dim=1)


def fmt(m):
    return (f"BIS={m['bis']:.4f} | precision={m['precision']:.4f} | "
            f"recall={m['recall']:.4f} | fpr={m['fpr']:.4f}")


def per_arch_recall(flags, is_triggered, arch_tags):
    for arch in ARCHS:
        mask = torch.tensor([a == arch for a in arch_tags]) & is_triggered
        if mask.sum():
            print(f"  {arch}: recall={flags[mask].float().mean().item():.4f} (n={mask.sum().item()})")
