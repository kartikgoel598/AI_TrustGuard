import os
import pandas as pd
import torch
from detection.rag_poison_detector.raw_document import (
    capture_activations,
    get_test_topics,
    get_layer_matrix,
    encode,
    rates,
    MODEL_DIR,
    POSITIVE_TYPES,
)  
from rag_extractor import RAG_LAYERS
from tier_specialist import load_sae, best_feature_for_tier, flag_with, TIERS
CLEANBASE_RESULTS_PATH = "cleanbase_evaluation_results.csv" 
def get_b_flags(records, test_ids, device):
    train_feats = {}
    test_feats = {}
    train_rows = None
    test_rows = None
 
    for layer_idx in RAG_LAYERS:
        model, mean, scale = load_sae(layer_idx, device)
 
        train_rows, train_x = get_layer_matrix(records, layer_idx, "original", test_ids, False)
        train_feats[layer_idx] = encode(model, mean, scale, train_x, device)
 
        test_rows, test_x = get_layer_matrix(records, layer_idx, "original", test_ids, True)
        test_feats[layer_idx] = encode(model, mean, scale, test_x, device)
 
    train_types = [r["attack_type"] for r in train_rows]
 
    picks = []
    for tier in TIERS:
        best = None
        for layer_idx in RAG_LAYERS:
            feat_idx, threshold, bis = best_feature_for_tier(train_feats[layer_idx], train_types, tier)
            if best is None or bis > best[3]:
                best = (layer_idx, feat_idx, threshold, bis)
        picks.append(best[:3])
        print(f"{tier} specialist: layer {best[0]} feature {best[1]} threshold {best[2]:.4f}")
 
    flagged = flag_with(test_feats, picks)
    return test_rows, flagged, picks
 
 
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    records = capture_activations(device)
    test_ids = get_test_topics(records)
 
    test_rows, b_flagged, picks = get_b_flags(records, test_ids, device)
 
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({"picks": picks}, os.path.join(MODEL_DIR, "activation_shift_config.pt"))
 
    a_df = pd.read_csv(CLEANBASE_RESULTS_PATH)[["document_id", "flagged"]].rename(columns={"flagged": "a_flagged"})
    b_df = pd.DataFrame({
        "document_id": [r["document_id"] for r in test_rows],
        "attack_type": [r["attack_type"] for r in test_rows],
        "b_flagged": b_flagged.tolist(),
    })
 
    merged = b_df.merge(a_df, on="document_id")
    print(f"\nheld-out documents: {len(b_df)}, matched with CleanBase results: {len(merged)}")
 
    is_pos = merged["attack_type"].isin(POSITIVE_TYPES).values
    a = merged["a_flagged"].values.astype(bool)
    b = merged["b_flagged"].values.astype(bool)
    either = a | b
 
    print("\noverall on held-out topics:")
    for name, flags in (("CleanBase alone", a), ("activation detector alone", b), ("CleanBase or activation", either)):
        recall, fpr, bis = rates(torch.tensor(flags[is_pos]), torch.tensor(flags[~is_pos]))
        print(f"  {name}: recall={recall:.3f} false alarm rate={fpr:.3f} bis={bis:.3f}")
 
    print("\nper attack type:")
    for attack in ("basic", "adaptive", "hard_negative", "clean"):
        mask = (merged["attack_type"] == attack).values
        missed = mask & ~a
        b_on_missed = b[missed].mean() if missed.sum() > 0 else float("nan")
        print(f"  {attack}: A={a[mask].mean():.3f} B={b[mask].mean():.3f} A or B={either[mask].mean():.3f} "
              f"| B on documents A missed={b_on_missed:.3f} (n={int(missed.sum())})")
 
 
if __name__ == "__main__":
    main()
 