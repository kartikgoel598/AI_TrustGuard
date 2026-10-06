import time 
import numpy as np 
import pandas as pd 

from detection.rag_poison_detector.clean_base_detector import CleanBaseDetector 
CORPUS_PATH = "dataset/rag_dataset/rag_corpus.csv"
METADATA_PATH = "dataset/rag_dataset/rag_metadata.csv"
PERCENTILES = [50, 60, 65, 70, 75, 80, 85, 90]

 
def main():
    np.random.seed(42)
 
    corpus = pd.read_csv(CORPUS_PATH)
    metadata = pd.read_csv(METADATA_PATH)
 
    documents = corpus["content"].tolist()
    document_ids = corpus["document_id"].tolist()
    meta = metadata.set_index("document_id").loc[document_ids]
    attack_types = meta["attack_type"].values
    is_poison = meta["is_poison"].values.astype(bool)
 
    detector = CleanBaseDetector(device="cuda")
    embeddings = detector.embedder.embed(documents)
    graph = detector.build_knn_graph(embeddings)
    print(f"graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
 
    for p in PERCENTILES:
        start = time.time()
        detector.prune_percentile = p
        pruned = detector.prune_graph(graph)
        flagged_indices, cliques = detector.find_flagged_cliques(pruned)
 
        flagged = np.zeros(len(documents), dtype=bool)
        for i in flagged_indices:
            flagged[i] = True
 
        tp = int((flagged & is_poison).sum())
        fn = int((~flagged & is_poison).sum())
        fp = int((flagged & ~is_poison).sum())
        tn = int((~flagged & ~is_poison).sum())
 
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
 
        print(f"\npercentile {p}: kept {pruned.number_of_edges()} edges, {len(cliques)} cliques, {time.time() - start:.0f}s")
        for attack in ("basic", "adaptive", "hard_negative", "clean"):
            mask = attack_types == attack
            rate = flagged[mask].mean()
            print(f"  {attack}: {rate * 100:.1f}%")
        print(f"  precision={precision:.3f} recall={recall:.3f} fpr={fpr:.4f}")
 
 
if __name__ == "__main__":
    main()