import numpy as np
import pandas as pd

from detection.rag_poison_detector.clean_base_detector import CleanBaseDetector

CORPUS_PATH = "dataset/rag_dataset/rag_corpus.csv"
CLUSTER_SIZE = 8
RATIOS = [0.02, 0.05, 0.10]
PERCENTILES = [90, 93, 95, 97, 98, 99]
SEEDS = [0, 1, 2]
POSITIVE_TYPES = ["basic", "adaptive"]


def pick_subset(corpus, ratio, seed):
    rng = np.random.RandomState(seed)
    is_poison = corpus["is_poison"].astype(bool).values
    non_poison_idx = np.where(~is_poison)[0]

    n_poison = int(round(ratio * len(non_poison_idx) / (1 - ratio)))
    n_queries = max(1, n_poison // CLUSTER_SIZE)

    query_ids = sorted(corpus["query_id"].unique())
    attacked = rng.choice(query_ids, size=n_queries, replace=False)

    keep_poison = []
    for q in attacked:
        tier = rng.choice(POSITIVE_TYPES)
        mask = ((corpus["query_id"] == q) & (corpus["attack_type"] == tier)).values
        keep_poison.extend(np.where(mask)[0].tolist())

    keep = np.concatenate([non_poison_idx, np.array(keep_poison, dtype=int)])
    keep.sort()
    return keep


def score(flagged, attack_types):
    is_pos = np.isin(attack_types, POSITIVE_TYPES)
    tp = int((flagged & is_pos).sum())
    fn = int((~flagged & is_pos).sum())
    fp = int((flagged & ~is_pos).sum())
    tn = int((~flagged & ~is_pos).sum())

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    bis = f1 * (1 - fpr)

    decoy_fp = int((flagged & (attack_types == "hard_negative")).sum())
    decoy_share = decoy_fp / fp if fp > 0 else float("nan")

    return {
        "recall": recall,
        "basic": rate(flagged, attack_types == "basic"),
        "adaptive": rate(flagged, attack_types == "adaptive"),
        "precision": precision,
        "clean": rate(flagged, attack_types == "clean"),
        "decoy": rate(flagged, attack_types == "hard_negative"),
        "decoy_share": decoy_share,
        "bis": bis,
        "share": is_pos.mean(),
        "size": len(attack_types),
    }


def rate(flagged, mask):
    if mask.sum() == 0:
        return float("nan")
    return flagged[mask].mean()


def main():
    corpus = pd.read_csv(CORPUS_PATH)
    documents = corpus["content"].tolist()
    all_types = corpus["attack_type"].values

    detector = CleanBaseDetector(device="cuda")
    embeddings = detector.embedder.embed(documents)
    print(f"embedded {len(documents)} documents")

    for ratio in RATIOS:
        results = {p: [] for p in PERCENTILES}

        for seed in SEEDS:
            np.random.seed(seed)
            keep = pick_subset(corpus, ratio, seed)
            types = all_types[keep]
            graph = detector.build_knn_graph(embeddings[keep])

            for p in PERCENTILES:
                detector.prune_percentile = p
                pruned = detector.prune_graph(graph)
                flagged_indices, cliques = detector.find_flagged_cliques(pruned)

                flagged = np.zeros(len(keep), dtype=bool)
                for i in flagged_indices:
                    flagged[i] = True

                results[p].append(score(flagged, types))

        first = results[PERCENTILES[0]]
        share = np.mean([r["share"] for r in first])
        size = np.mean([r["size"] for r in first])
        print(f"\ntarget poison share {ratio:.0%}: actual {share:.1%}, about {size:.0f} documents, {len(SEEDS)} runs averaged")

        keys = ("recall", "basic", "adaptive", "precision", "clean", "decoy", "decoy_share", "bis")
        for p in PERCENTILES:
            avg = {k: np.nanmean([r[k] for r in results[p]]) for k in keys}
            print(f"  percentile {p}: recall={avg['recall']:.3f} (basic {avg['basic']:.3f}, adaptive {avg['adaptive']:.3f}) "
                  f"precision={avg['precision']:.3f} clean flagged={avg['clean']:.3f} decoy flagged={avg['decoy']:.3f} "
                  f"decoys share of false alarms={avg['decoy_share']:.2f} bis={avg['bis']:.3f}")


if __name__ == "__main__":
    main()