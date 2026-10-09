import argparse
import time

import numpy as np
import pandas as pd
import torch

from detection.rag_poison_detector.product.clean_base_detector import CleanBaseDetector
from detection.rag_poison_detector.product.activation_shift_detector import load_sae, flag_with, CONFIG_PATH, encode
from detection.rag_poison_detector.product.pipeline import run_layer2
from detection.rag_poison_detector.training.train_activation_sae import (
    capture_activations, get_test_topics, CORPUS_PATH, POSITIVE_TYPES,
)

SHARES = (0.02, 0.05, 0.10)
PERCENTILES = (95, 97, 98)
SEEDS = (0, 1, 2)
N_REALISTIC = 500


# ---------- shared helpers ----------

def rate(flagged, mask):
    return float(flagged[mask].mean()) if mask.any() else float("nan")


def metrics(flagged, types):
    pos = np.isin(types, POSITIVE_TYPES)
    tp = int((flagged & pos).sum())
    return {
        "recall": tp / pos.sum() if pos.any() else float("nan"),
        "adaptive_recall": rate(flagged, types == "adaptive"),
        "precision": tp / flagged.sum() if flagged.any() else float("nan"),
        "clean_flag_rate": rate(flagged, types == "clean"),
        "decoy_flag_rate": rate(flagged, types == "hard_negative"),
    }


def show(label, m):
    print(f"{label:<34}" + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))


def load_data(device):
    records = capture_activations(device)
    originals = [r for r in records if r["variant"] == "original"]
    corpus = pd.read_csv(CORPUS_PATH).set_index("document_id").loc[[r["document_id"] for r in originals]]
    return records, originals, corpus["content"].tolist()


def activation_flags(rows, device):
    picks = torch.load(CONFIG_PATH)["picks"]
    feats = {}
    for layer in sorted(set(p[0] for p in picks)):
        model, mean, scale = load_sae(layer, device)
        x = torch.stack([r["acts"][layer] for r in rows]).float()
        feats[layer] = encode(model, mean, scale, x, device)
    return flag_with(feats, picks).numpy()


def embed_all(documents, device):
    return CleanBaseDetector(device=device).embed(documents)


def clean_base_flags(embeddings, percentiles, seed):
    # the kNN graph is built once and reused for every percentile
    np.random.seed(seed)
    detector = CleanBaseDetector.__new__(CleanBaseDetector)
    detector.prune_method = "percentile"
    graph = detector.build_knn_graph(embeddings)
    out = {}
    for p in percentiles:
        detector.prune_percentile = p
        flagged, _ = detector.find_flagged_cliques(detector.prune_graph(graph))
        mask = np.zeros(len(embeddings), dtype=bool)
        mask[list(flagged)] = True
        out[p] = mask
    return out


def make_subset(types, base_ids, n, share, rng, min_poison=0):
    # whole topics are kept together so poison clusters and their neighbours stay intact
    n_poison = int(round(n * share))
    if n_poison > 0:
        n_poison = max(n_poison, min_poison)
    pos_mask = np.isin(types, POSITIVE_TYPES)

    def pick(mask, count):
        topics = {}
        for i in np.where(mask)[0]:
            topics.setdefault(base_ids[i], []).append(i)
        keys = list(topics)
        rng.shuffle(keys)
        chosen = []
        for k in keys:
            if len(chosen) >= count:
                break
            chosen.extend(topics[k])
        return chosen[:count]

    idx = pick(pos_mask, n_poison) + pick(~pos_mask, n - n_poison)
    return np.array(sorted(idx))


# ---------- modes ----------

def mode_heldout(device):
    records, originals, _ = load_data(device)
    test_ids = get_test_topics(records)
    types_all = lambda rows: np.array([r["attack_type"] for r in rows])

    test = [r for r in originals if r["base_document_id"] in test_ids]
    train = [r for r in originals if r["base_document_id"] not in test_ids]
    ablated = [r for r in records if r["variant"] == "ablated" and r["base_document_id"] in test_ids]
    test_types = types_all(test)

    show("activation detector", metrics(activation_flags(test, device), test_types))

    abl_flags = activation_flags(ablated, device)
    print(f"marker removed: recall on {len(ablated)} held-out poisoned docs = {abl_flags.mean():.3f}")

    show("baseline: marker word", metrics(np.array([r["has_marker"] for r in test]), test_types))

    train_neg_len = np.array([r["content_length"] for r in train if r["attack_type"] not in POSITIVE_TYPES])
    cutoff = np.percentile(train_neg_len, 95)
    show("baseline: long document", metrics(np.array([r["content_length"] > cutoff for r in test]), test_types))


def mode_realistic(device):
    _, originals, docs = load_data(device)
    types = np.array([r["attack_type"] for r in originals])
    base_ids = np.array([r["base_document_id"] for r in originals])
    act = activation_flags(originals, device)
    emb = embed_all(docs, device)

    for share in SHARES:
        print(f"\n=== {share:.0%} poison, {N_REALISTIC} documents ===")
        acc = {}
        for seed in SEEDS:
            idx = make_subset(types, base_ids, N_REALISTIC, share, np.random.RandomState(seed))
            cb = clean_base_flags(emb[idx], PERCENTILES, seed)
            for p in PERCENTILES:
                for name, f in (("cleanbase", cb[p]), ("activation", act[idx]),
                                ("either", cb[p] | act[idx]), ("both", cb[p] & act[idx])):
                    acc.setdefault((p, name), []).append(metrics(f, types[idx]))
        for (p, name), runs in acc.items():
            show(f"p{p} {name}", {k: np.nanmean([r[k] for r in runs]) for k in runs[0]})


def mode_prevalence(device):
    _, originals, docs = load_data(device)
    types = np.array([r["attack_type"] for r in originals])
    base_ids = np.array([r["base_document_id"] for r in originals])
    emb = embed_all(docs, device)
    percentiles = range(90, 100)

    for share in SHARES:
        print(f"\n=== CleanBase only, {share:.0%} poison ===")
        acc = {p: [] for p in percentiles}
        for seed in SEEDS:
            idx = make_subset(types, base_ids, N_REALISTIC, share, np.random.RandomState(seed))
            for p, f in clean_base_flags(emb[idx], percentiles, seed).items():
                acc[p].append(metrics(f, types[idx]))
        for p, runs in acc.items():
            show(f"percentile {p}", {k: np.nanmean([r[k] for r in runs]) for k in runs[0]})


def mode_size(device):
    _, originals, docs = load_data(device)
    types = np.array([r["attack_type"] for r in originals])
    base_ids = np.array([r["base_document_id"] for r in originals])
    emb = embed_all(docs, device)

    for share in (0.0, 0.05):
        print(f"\n=== CleanBase, percentile 97, {share:.0%} poison ===")
        for n in (20, 50, 100, 200, 500, 1000, 2000):
            if n > len(types):
                continue
            runs = []
            for seed in range(5):
                # one attack is a cluster of 8 poisoned documents, so small collections get at least 8
                idx = make_subset(types, base_ids, n, share, np.random.RandomState(seed), min_poison=8)
                runs.append(metrics(clean_base_flags(emb[idx], (97,), seed)[97], types[idx]))
            show(f"n={n}", {k: np.nanmean([r[k] for r in runs]) for k in runs[0]})


def mode_false_alarm(device):
    _, originals, docs = load_data(device)
    types = np.array([r["attack_type"] for r in originals])
    base_ids = np.array([r["base_document_id"] for r in originals])
    act = activation_flags(originals, device)
    emb = embed_all(docs, device)

    print("clean collections (no poison): share of collections where at least k documents are flagged by both")
    for n in (50, 100, 200, 500, 1000):
        if n > (~np.isin(types, POSITIVE_TYPES)).sum():
            continue
        counts, shares = [], []
        for seed in range(20):
            idx = make_subset(types, base_ids, n, 0.0, np.random.RandomState(seed))
            both = clean_base_flags(emb[idx], (97,), seed)[97] & act[idx]
            counts.append(int(both.sum()))
            shares.append(both.mean())
        counts = np.array(counts)
        rule = "  ".join(f"k>={k}: {(counts >= k).mean():.0%}" for k in (1, 2, 3, 5))
        print(f"n={n:<5} both-share={np.mean(shares):.3f}  {rule}")


def mode_full_corpus(device):
    _, originals, docs = load_data(device)
    types = np.array([r["attack_type"] for r in originals])
    emb = embed_all(docs, device)
    for p, f in clean_base_flags(emb, range(90, 100), 0).items():
        show(f"full corpus percentile {p}", metrics(f, types))


def mode_demo(device):
    _, originals, docs = load_data(device)
    types = np.array([r["attack_type"] for r in originals])
    base_ids = np.array([r["base_document_id"] for r in originals])

    for n in (20, 200):
        rng = np.random.RandomState(7)
        idx = make_subset(types, base_ids, n, 0.05, rng)
        rng.shuffle(idx)
        names = [f"doc_{i + 1:03d}.txt" for i in range(len(idx))]
        truth = {names[i]: types[j] for i, j in enumerate(idx)}

        start = time.time()
        result = run_layer2([docs[j] for j in idx], file_names=names, device=device)
        detail = result.detail
        print(f"\n=== {n} documents, {time.time() - start:.1f}s, status {result.status} ===")
        for w in detail["warnings"]:
            print("warning:", w)
        print("high confidence:", [(f, truth[f]) for f in detail["high_confidence_files"]])
        print("needs review   :", len(detail["review_files"]), "files")
        print("actually poisoned:", sorted(f for f, t in truth.items() if t in POSITIVE_TYPES))

        pd.DataFrame({"file": names, "true_type": [truth[f] for f in names]}).to_csv(
            f"demo_answer_key_{n}.csv", index=False)


MODES = {
    "heldout": mode_heldout,
    "realistic": mode_realistic,
    "prevalence": mode_prevalence,
    "size": mode_size,
    "false-alarm": mode_false_alarm,
    "full-corpus": mode_full_corpus,
    "demo": mode_demo,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=MODES)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODES[args.mode](device)


if __name__ == "__main__":
    main()