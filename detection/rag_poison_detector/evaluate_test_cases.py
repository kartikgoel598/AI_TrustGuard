import numpy as np
import pandas as pd

from detection.rag_poison_detector.clean_base_detector import CleanBaseDetector

CORPUS_PATH = "dataset/rag_dataset/rag_corpus.csv"
METADATA_PATH = "dataset/rag_dataset/rag_metadata.csv"
PRUNE_PERCENTILE = 70


def main():
    corpus = pd.read_csv(CORPUS_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    documents = corpus["content"].tolist()
    document_ids = corpus["document_id"].tolist()

    np.random.seed(42)
    print(f"running CleanBaseDetector on {len(documents)} documents, prune percentile {PRUNE_PERCENTILE}")

    device = "cuda"
    detector = CleanBaseDetector(device=device, prune_percentile=PRUNE_PERCENTILE)
    result = detector.run({"documents": documents})

    if result.status == "error":
        print(f"ERROR: {result.detail['error']}")
        return

    print(f"total flagged: {result.detail['num_flagged']}/{len(documents)}")
    print(f"cliques found: {len(result.detail['cliques'])}")

    flagged_by_index = {d["index"]: d["flagged"] for d in result.detail["per_document"]}

    results_df = pd.DataFrame({
        "document_id": document_ids,
        "flagged": [flagged_by_index[i] for i in range(len(document_ids))],
    })

    merged = results_df.merge(metadata[["document_id", "attack_type", "is_poison"]], on="document_id")

    print("\nper attack_type breakdown:")
    for attack_type in merged["attack_type"].unique():
        subset = merged[merged["attack_type"] == attack_type]
        flagged_count = subset["flagged"].sum()
        total = len(subset)
        print(f"  {attack_type}: {flagged_count}/{total} flagged ({flagged_count / total * 100:.1f}%)")

    tp = len(merged[(merged["is_poison"] == True) & (merged["flagged"] == True)])
    fn = len(merged[(merged["is_poison"] == True) & (merged["flagged"] == False)])
    fp = len(merged[(merged["is_poison"] == False) & (merged["flagged"] == True)])
    tn = len(merged[(merged["is_poison"] == False) & (merged["flagged"] == False)])

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    bis = f1 * (1 - fpr)

    print(f"\noverall: precision={precision:.4f} recall={recall:.4f} fpr={fpr:.4f} f1={f1:.4f} bis={bis:.4f}")
    print(f"tp={tp} fn={fn} fp={fp} tn={tn}")

    results_df.merge(metadata, on="document_id").to_csv("cleanbase_evaluation_results.csv", index=False)
    print("\nfull results saved to cleanbase_evaluation_results.csv")


if __name__ == "__main__":
    main()