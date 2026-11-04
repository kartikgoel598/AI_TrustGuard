import os
import numpy as np
import pandas as pd
import torch
 
from raw_document import capture_activations, get_test_topics, get_layer_matrix, encode, MODEL_DIR
from tier_specialist import load_sae, flag_with
from detection.rag_poison_detector.clean_base_detector import CleanBaseDetector
 
CORPUS_PATH = "dataset/rag_dataset/rag_corpus.csv"
CLUSTER_SIZE = 8
RATIOS = [0.02, 0.05, 0.10]
PERCENTILES = [95, 97, 98]
SEEDS = [0, 1, 2]
POSITIVE_TYPES = ["basic", "adaptive"]
KEYS = ("recall", "basic", "adaptive", "precision", "clean", "decoy", "bis")

def rate(flagged, mask):
    if mask.sum() == 0:
        return float("nan")
    return flagged[mask].mean()
 
 
def metrics(flagged, types):
    is_pos = np.isin(types, POSITIVE_TYPES)
    tp = int((flagged & is_pos).sum())
    fn = int((~flagged & is_pos).sum())
    fp = int((flagged & ~is_pos).sum())
    tn = int((~flagged & ~is_pos).sum())
 
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    bis = f1 * (1 - fpr)
 
    return {
        "recall": recall,
        "basic": rate(flagged, types == "basic"),
        "adaptive": rate(flagged, types == "adaptive"),
        "precision": precision,
        "clean": rate(flagged, types == "clean"),
        "decoy": rate(flagged, types == "hard_negative"),
        "bis": bis,
    }
 
 
def get_b_flags(records, device):
    config = torch.load(os.path.join(MODEL_DIR, "activation_shift_config.pt"))
    picks = config["picks"]
    layers = sorted(set(p[0] for p in picks))
 
    feats_by_layer = {}
    rows = None
    for layer_idx in layers:
        model, mean, scale = load_sae(layer_idx, device)
        rows, x = get_layer_matrix(records, layer_idx, "original", set(), False)
        feats_by_layer[layer_idx] = encode(model, mean, scale, x, device)
 
    flagged = flag_with(feats_by_layer, picks).numpy()
    ids = [r["document_id"] for r in rows]
    return dict(zip(ids, flagged.tolist()))
 
 
def pick_subset(corpus, ratio, seed, heldout_queries):
    rng = np.random.RandomState(seed)
    is_poison = corpus["is_poison"].astype(bool).values
    non_poison_idx = np.where(~is_poison)[0]
 
    n_poison = int(round(ratio * len(non_poison_idx) / (1 - ratio)))
    n_queries = min(max(1, n_poison // CLUSTER_SIZE), len(heldout_queries))
    attacked = rng.choice(heldout_queries, size=n_queries, replace=False)
 
    keep_poison = []
    for q in attacked:
        tier = rng.choice(POSITIVE_TYPES)
        mask = ((corpus["query_id"] == q) & (corpus["attack_type"] == tier)).values
        keep_poison.extend(np.where(mask)[0].tolist())
 
    keep = np.concatenate([non_poison_idx, np.array(keep_poison, dtype=int)])
    keep.sort()
    return keep
 
 
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    corpus = pd.read_csv(CORPUS_PATH)
    documents = corpus["content"].tolist()
    doc_ids = corpus["document_id"].tolist()
    all_types = corpus["attack_type"].values
 
    records = capture_activations(device)
    test_ids = get_test_topics(records)
    heldout_queries = sorted(test_ids)
 
    b_by_id = get_b_flags(records, device)
    b_all = np.array([b_by_id[d] for d in doc_ids], dtype=bool)
 
    detector = CleanBaseDetector(device=device)
    embeddings = detector.embedder.embed(documents)
    print(f"embedded {len(documents)} documents, {len(heldout_queries)} held-out queries")
 
    for ratio in RATIOS:
        results = {}
 
        for seed in SEEDS:
            np.random.seed(seed)
            keep = pick_subset(corpus, ratio, seed, heldout_queries)
            types = all_types[keep]
            b = b_all[keep]
 
            results.setdefault("activation detector alone", []).append(metrics(b, types))
            graph = detector.build_knn_graph(embeddings[keep])
 
            for p in PERCENTILES:
                detector.prune_percentile = p
                pruned = detector.prune_graph(graph)
                flagged_indices, cliques = detector.find_flagged_cliques(pruned)
 
                a = np.zeros(len(keep), dtype=bool)
                for i in flagged_indices:
                    a[i] = True
 
                results.setdefault(f"CleanBase p{p}", []).append(metrics(a, types))
                results.setdefault(f"either flags it p{p}", []).append(metrics(a | b, types))
                results.setdefault(f"both flag it p{p}", []).append(metrics(a & b, types))
 
        print(f"\ntarget poison share {ratio:.0%}, {len(SEEDS)} runs averaged")
        for name, runs in results.items():
            avg = {k: np.nanmean([r[k] for r in runs]) for k in KEYS}
            print(f"  {name}: recall={avg['recall']:.3f} (basic {avg['basic']:.3f}, adaptive {avg['adaptive']:.3f}) "
                  f"precision={avg['precision']:.3f} clean flagged={avg['clean']:.3f} "
                  f"decoy flagged={avg['decoy']:.3f} bis={avg['bis']:.3f}")
 
 
if __name__ == "__main__":
    main()
 