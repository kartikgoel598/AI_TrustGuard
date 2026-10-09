import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model_loader import load_model_from_registry
from detection.diff_sae.model import DiffSAE, compute_loss
from detection.rag_poison_detector.product.rag_extractor import RagActivationExtractor, RAG_LAYERS
from detection.rag_poison_detector.product.activation_shift_detector import encode, MODEL_DIR, CONFIG_PATH

CORPUS_PATH = "dataset/rag_dataset/rag_corpus.csv"
BASELINE_MODEL = "smollm2_360m_benign_full-rank"
CACHE_PATH = "activation_capture/rag_diff/rag_raw_activations.pt"

MARKER = "TRUSTGUARD_POISON_TARGET"
REPLACEMENT = "Zanzibar"
POSITIVE_TYPES = ("basic", "adaptive")
NEGATIVE_TYPES = ("clean", "hard_negative")
TIERS = ("basic", "adaptive")

TEST_FRACTION = 0.2
SEED = 42
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
SPARSITY_LAMBDA = 0.5
NUM_EPOCHS = 100


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

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
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


def dead_feature_fraction(feats):
    ever_active = (feats > 0).any(dim=0)
    return 1.0 - ever_active.float().mean().item()


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true", help="retrain and replace the saved SAEs and picks")
    args = parser.parse_args()

    if os.path.exists(CONFIG_PATH) and not args.overwrite:
        print(f"{CONFIG_PATH} already exists. Pass --overwrite to retrain and replace the saved models.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)

    records = capture_activations(device)
    test_ids = get_test_topics(records)
    print(f"{len(test_ids)} topics held out for testing")

    os.makedirs(MODEL_DIR, exist_ok=True)

    train_feats = {}
    train_rows = None
    for layer_idx in RAG_LAYERS:
        train_rows, train_x = get_layer_matrix(records, layer_idx, "original", test_ids, False)
        print(f"\nlayer {layer_idx}: training on {len(train_rows)} documents")

        model, mean, scale = train_sae(train_x, device)
        train_feats[layer_idx] = encode(model, mean, scale, train_x, device)
        print(f"dead feature fraction: {dead_feature_fraction(train_feats[layer_idx]):.2%}")

        torch.save({
            "state_dict": model.state_dict(),
            "mean": mean,
            "scale": scale,
            "layer_idx": layer_idx,
        }, os.path.join(MODEL_DIR, f"raw_sae_layer{layer_idx}.pt"))

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

    torch.save({"picks": picks}, CONFIG_PATH)
    print(f"saved picks to {CONFIG_PATH}")


if __name__ == "__main__":
    main()