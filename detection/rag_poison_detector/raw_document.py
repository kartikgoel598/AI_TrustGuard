import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model_loader import load_model_from_registry
from rag_extractor import RagActivationExtractor, RAG_LAYERS
from detection.diff_sae.model import DiffSAE, compute_loss

CORPUS_PATH = "dataset/rag_dataset/rag_corpus.csv"
BASELINE_MODEL = "smollm2_360m_benign_full-rank"
CACHE_PATH = "activation_capture/rag_diff/rag_raw_activations.pt"
MODEL_DIR = "detection/rag_poison_detector/trained_models"

MARKER = "TRUSTGUARD_POISON_TARGET"
REPLACEMENT = "Zanzibar"
POSITIVE_TYPES = ("basic", "adaptive")

TEST_FRACTION = 0.2
SEED = 42
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
SPARSITY_LAMBDA = 0.5
NUM_EPOCHS = 100
TOP_K = 5


def capture_activations(device):
    if os.path.exists(CACHE_PATH):
        print(f"loading cached activations from {CACHE_PATH}")
        return torch.load(CACHE_PATH)

    corpus = pd.read_csv(CORPUS_PATH)
    baseline = load_model_from_registry(BASELINE_MODEL)
    extractor = RagActivationExtractor(baseline["model"], baseline["tokenizer"], device)

    records = []
    for i, row in corpus.iterrows():
        content = row["content"]
        records.append({
            "document_id": row["document_id"],
            "base_document_id": row["query_id"],
            "attack_type": row["attack_type"],
            "variant": "original",
            "content_length": len(content),
            "has_marker": MARKER in content,
            "acts": extractor.extract_for_document(content),
        })

        if row["attack_type"] in POSITIVE_TYPES:
            ablated = content.replace(MARKER, REPLACEMENT)
            records.append({
                "document_id": row["document_id"],
                "base_document_id": row["query_id"],
                "attack_type": row["attack_type"],
                "variant": "ablated",
                "content_length": len(ablated),
                "has_marker": MARKER in ablated,
                "acts": extractor.extract_for_document(ablated),
            })

        if (i + 1) % 200 == 0:
            print(f"captured {i + 1}/{len(corpus)} documents")

    torch.save(records, CACHE_PATH)
    return records


def get_test_topics(records):
    base_ids = sorted(set(r["base_document_id"] for r in records))
    rng = np.random.RandomState(SEED)
    rng.shuffle(base_ids)
    n_test = int(len(base_ids) * TEST_FRACTION)
    return set(base_ids[:n_test])


def get_layer_matrix(records, layer_idx, variant, test_ids, want_test):
    rows = []
    for r in records:
        if r["variant"] != variant:
            continue
        in_test = r["base_document_id"] in test_ids
        if in_test == want_test:
            rows.append(r)
    x = torch.stack([r["acts"][layer_idx] for r in rows]).float()
    return rows, x


def train_sae(train_x, device):
    mean = train_x.mean(dim=0)
    centered = train_x - mean
    avg_sq_norm = (centered ** 2).sum(dim=1).mean()
    scale = (centered.shape[1] / avg_sq_norm).sqrt()
    data = centered * scale

    model = DiffSAE(input_dim=data.shape[1], expansion_factor=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(NUM_EPOCHS):
        for batch in loader:
            batch = batch.to(device)
            recon, feats = model(batch)
            loss, _, _ = compute_loss(batch, recon, feats, sparsity_lambda=SPARSITY_LAMBDA)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.renormalize_decoder()

    model.eval()
    return model, mean, scale


def encode(model, mean, scale, x, device):
    with torch.no_grad():
        return model.encode(((x - mean) * scale).to(device)).cpu()


def select_features(feats, is_pos, k):
    pos = feats[is_pos]
    neg = feats[~is_pos]

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

    top_idx = torch.argsort(bis, descending=True)[:k]
    return top_idx.tolist(), thresholds[top_idx]


def rates(pos_flagged, neg_flagged):
    tp = pos_flagged.sum().item()
    fn = len(pos_flagged) - tp
    fp = neg_flagged.sum().item()
    tn = len(neg_flagged) - fp

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    bis = f1 * (1 - fpr)
    return recall, fpr, bis


def run_layer(records, layer_idx, test_ids, device):
    print(f"\nlayer {layer_idx}")

    train_rows, train_x = get_layer_matrix(records, layer_idx, "original", test_ids, False)
    print(f"training on {len(train_rows)} documents")
    model, mean, scale = train_sae(train_x, device)

    train_feats = encode(model, mean, scale, train_x, device)
    train_pos = torch.tensor([r["attack_type"] in POSITIVE_TYPES for r in train_rows])
    feat_idxs, thresholds = select_features(train_feats, train_pos, TOP_K)

    test_rows, test_x = get_layer_matrix(records, layer_idx, "original", test_ids, True)
    test_feats = encode(model, mean, scale, test_x, device)
    test_pos = torch.tensor([r["attack_type"] in POSITIVE_TYPES for r in test_rows])

    abl_rows, abl_x = get_layer_matrix(records, layer_idx, "ablated", test_ids, True)
    abl_feats = encode(model, mean, scale, abl_x, device)

    print(f"held-out topics: {len(test_rows)} documents, {len(abl_rows)} marker-removed copies")
    print(f"top feature for this layer: index={feat_idxs[0]} threshold={thresholds[0].item():.6f}")

    for k in (1, TOP_K):
        idxs = feat_idxs[:k]
        thr = thresholds[:k]

        def flag(feats):
            return (feats[:, idxs] > thr).any(dim=1)

        neg_flagged = flag(test_feats[~test_pos])
        recall_orig, fpr, bis_orig = rates(flag(test_feats[test_pos]), neg_flagged)
        recall_abl, _, bis_abl = rates(flag(abl_feats), neg_flagged)

        print(f"  top {k} features: original poison recall={recall_orig:.3f} bis={bis_orig:.3f} | "
              f"marker removed recall={recall_abl:.3f} bis={bis_abl:.3f} | false alarm rate={fpr:.3f}")

        print(f"    per attack_type flag rate (top {k}):")
        for attack in ("basic", "adaptive", "hard_negative", "clean"):
            mask = torch.tensor([r["attack_type"] == attack for r in test_rows])
            if mask.sum() == 0:
                continue
            rate = flag(test_feats[mask]).float().mean().item()
            print(f"      {attack}: {rate:.3f} (n={mask.sum().item()})")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "mean": mean,
        "scale": scale,
        "feature_idxs": feat_idxs,
        "thresholds": thresholds,
        "layer_idx": layer_idx,
    }, os.path.join(MODEL_DIR, f"raw_sae_layer{layer_idx}.pt"))

    return feat_idxs, thresholds, test_rows, test_feats, abl_feats


def run_baselines(records, test_ids):
    print("\nsimple rules on held-out topics")

    train_neg_lengths = [
        r["content_length"] for r in records
        if r["variant"] == "original" and r["base_document_id"] not in test_ids
        and r["attack_type"] not in POSITIVE_TYPES
    ]
    length_threshold = np.percentile(train_neg_lengths, 95)

    test_orig = [r for r in records if r["variant"] == "original" and r["base_document_id"] in test_ids]
    test_abl = [r for r in records if r["variant"] == "ablated" and r["base_document_id"] in test_ids]
    pos_orig = [r for r in test_orig if r["attack_type"] in POSITIVE_TYPES]
    neg = [r for r in test_orig if r["attack_type"] not in POSITIVE_TYPES]

    rules = [
        ("marker word rule", lambda r: r["has_marker"]),
        (f"length rule (> {length_threshold:.0f} characters)", lambda r: r["content_length"] > length_threshold),
    ]

    for name, rule in rules:
        recall_orig = np.mean([rule(r) for r in pos_orig])
        recall_abl = np.mean([rule(r) for r in test_abl])
        fpr = np.mean([rule(r) for r in neg])
        print(f"  {name}: original poison recall={recall_orig:.3f} | marker removed recall={recall_abl:.3f} | false alarm rate={fpr:.3f}")


def run_combined(feat_idxs_by_layer, thresholds_by_layer, all_test_feats, all_abl_feats, all_test_rows):
    l14_feat, l14_thr = feat_idxs_by_layer[14][0], thresholds_by_layer[14][0]
    l26_feat, l26_thr = feat_idxs_by_layer[26][0], thresholds_by_layer[26][0]

    l14_test_feats = all_test_feats[14]
    l26_test_feats = all_test_feats[26]
    l14_abl_feats = all_abl_feats[14]
    l26_abl_feats = all_abl_feats[26]

    test_rows_ref = all_test_rows[14]
    test_pos_ref = torch.tensor([r["attack_type"] in POSITIVE_TYPES for r in test_rows_ref])

    def combined_flag(l14_feats, l26_feats):
        return (l14_feats[:, l14_feat] > l14_thr) | (l26_feats[:, l26_feat] > l26_thr)

    flagged_orig = combined_flag(l14_test_feats, l26_test_feats)
    flagged_abl = combined_flag(l14_abl_feats, l26_abl_feats)

    neg_flagged = flagged_orig[~test_pos_ref]
    recall_orig, fpr, bis_orig = rates(flagged_orig[test_pos_ref], neg_flagged)
    recall_abl, _, bis_abl = rates(flagged_abl, neg_flagged)

    print(f"\ncombined layer14-top1 + layer26-top1:")
    print(f"  original poison recall={recall_orig:.3f} bis={bis_orig:.3f} | "
          f"marker removed recall={recall_abl:.3f} bis={bis_abl:.3f} | false alarm rate={fpr:.3f}")

    print("  per attack_type flag rate:")
    for attack in ("basic", "adaptive", "hard_negative", "clean"):
        mask = torch.tensor([r["attack_type"] == attack for r in test_rows_ref])
        if mask.sum() == 0:
            continue
        rate = flagged_orig[mask].float().mean().item()
        print(f"    {attack}: {rate:.3f} (n={mask.sum().item()})")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    print(f"seed {SEED}, cache file: {CACHE_PATH} (exists: {os.path.exists(CACHE_PATH)})")

    records = capture_activations(device)
    test_ids = get_test_topics(records)
    print(f"{len(test_ids)} topics held out for testing")

    run_baselines(records, test_ids)

    feat_idxs_by_layer = {}
    thresholds_by_layer = {}
    all_test_feats = {}
    all_abl_feats = {}
    all_test_rows = {}

    for layer_idx in RAG_LAYERS:
        feat_idxs, thresholds, test_rows, test_feats, abl_feats = run_layer(records, layer_idx, test_ids, device)
        feat_idxs_by_layer[layer_idx] = feat_idxs
        thresholds_by_layer[layer_idx] = thresholds
        all_test_feats[layer_idx] = test_feats
        all_abl_feats[layer_idx] = abl_feats
        all_test_rows[layer_idx] = test_rows

    run_combined(feat_idxs_by_layer, thresholds_by_layer, all_test_feats, all_abl_feats, all_test_rows)


if __name__ == "__main__":
    main()