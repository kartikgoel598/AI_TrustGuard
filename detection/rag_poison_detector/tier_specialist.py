import os
import numpy as np
import torch

from raw_document import (
    capture_activations,
    get_test_topics,
    get_layer_matrix,
    encode,
    rates,
    MODEL_DIR,
    POSITIVE_TYPES,
)
from rag_extractor import RAG_LAYERS
from detection.diff_sae.model import DiffSAE

TIERS = ("basic", "adaptive")
NEGATIVE_TYPES = ("clean", "hard_negative")


def load_sae(layer_idx, device):
    checkpoint = torch.load(os.path.join(MODEL_DIR, f"raw_sae_layer{layer_idx}.pt"))
    model = DiffSAE(input_dim=960, expansion_factor=4).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["mean"], checkpoint["scale"]


def best_feature_for_tier(feats, attack_types, tier):
    keep = [i for i, a in enumerate(attack_types) if a == tier or a in NEGATIVE_TYPES]
    sub = feats[keep]
    is_pos = torch.tensor([attack_types[i] == tier for i in keep])
    pos = sub[is_pos]
    neg = sub[~is_pos]

    thresholds = torch.quantile(neg, 0.95, dim=0)
    tp = (pos > thresholds).sum(dim=0).float()
    fn = pos.shape[0] - tp
    fp = (neg > thresholds).sum(dim=0).float()
    tn = neg.shape[0] - fp

    recall = tp / (tp + fn)
    precision = tp / (tp + fp).clamp(min=1)
    fpr = fp / (fp + tn)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-9)
    bis = f1 * (1 - fpr)

    best = int(torch.argmax(bis))
    return best, thresholds[best].item(), bis[best].item()


def flag_with(feats_by_layer, picks):
    flagged = None
    for layer_idx, feat_idx, threshold in picks:
        current = feats_by_layer[layer_idx][:, feat_idx] > threshold
        flagged = current if flagged is None else (flagged | current)
    return flagged


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records = capture_activations(device)
    test_ids = get_test_topics(records)

    train_feats = {}
    test_feats = {}
    abl_feats = {}
    train_rows = test_rows = abl_rows = None

    for layer_idx in RAG_LAYERS:
        model, mean, scale = load_sae(layer_idx, device)

        train_rows, train_x = get_layer_matrix(records, layer_idx, "original", test_ids, False)
        train_feats[layer_idx] = encode(model, mean, scale, train_x, device)

        test_rows, test_x = get_layer_matrix(records, layer_idx, "original", test_ids, True)
        test_feats[layer_idx] = encode(model, mean, scale, test_x, device)

        abl_rows, abl_x = get_layer_matrix(records, layer_idx, "ablated", test_ids, True)
        abl_feats[layer_idx] = encode(model, mean, scale, abl_x, device)

    train_types = [r["attack_type"] for r in train_rows]

    picks = []
    for tier in TIERS:
        best = None
        for layer_idx in RAG_LAYERS:
            feat_idx, threshold, bis = best_feature_for_tier(train_feats[layer_idx], train_types, tier)
            if best is None or bis > best[3]:
                best = (layer_idx, feat_idx, threshold, bis)
        picks.append(best[:3])
        print(f"{tier} specialist: layer {best[0]} feature {best[1]} threshold {best[2]:.4f} train bis {best[3]:.3f}")

    test_types = np.array([r["attack_type"] for r in test_rows])
    abl_types = np.array([r["attack_type"] for r in abl_rows])

    flagged = flag_with(test_feats, picks)
    flagged_abl = flag_with(abl_feats, picks)

    is_pos = torch.tensor([t in POSITIVE_TYPES for t in test_types])
    neg_flagged = flagged[~is_pos]
    recall, fpr, bis = rates(flagged[is_pos], neg_flagged)
    recall_abl, _, bis_abl = rates(flagged_abl, neg_flagged)

    print(f"\nheld-out topics: {len(test_rows)} documents, {len(abl_rows)} marker-removed copies")
    print(f"original poison recall={recall:.3f} bis={bis:.3f} | marker removed recall={recall_abl:.3f} bis={bis_abl:.3f} | false alarm rate={fpr:.3f}")

    print("flag rate per attack type (original text):")
    for attack in ("basic", "adaptive", "hard_negative", "clean"):
        mask = torch.tensor(test_types == attack)
        print(f"  {attack}: {flagged[mask].float().mean().item():.3f} (n={int(mask.sum())})")

    print("flag rate per attack type (marker removed):")
    for attack in TIERS:
        mask = torch.tensor(abl_types == attack)
        print(f"  {attack}: {flagged_abl[mask].float().mean().item():.3f} (n={int(mask.sum())})")


if __name__ == "__main__":
    main()