import pandas as pd
import torch
 
from model_loader import load_model_from_registry
from rag_extractor import RagActivationExtractor, RAG_LAYERS

CORPUS_PATH = "dataset/rag_dataset/rag_corpus.csv"
BASELINE_MODEL = "smollm2_360m_benign_full-rank"
OUTPUT_PATH = "activation_capture/rag_diff/rag_diff_data.pt"

def main():
    corpus = pd.read_csv(CORPUS_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    baseline = load_model_from_registry(BASELINE_MODEL)
    model, tokenizer = baseline["model"], baseline["tokenizer"]
    extractor = RagActivationExtractor(model, tokenizer, device)
    grouped = corpus.groupby("pair_id")
    all_rows = []
 
    for i, (pair_id, group) in enumerate(grouped):
        clean_row = group[group["attack_type"] == "none"]
        if len(clean_row) == 0:
            continue
        clean_row = clean_row.iloc[0]
 
        clean_acts = extractor.extract_for_document(clean_row["content"])
 
        poisoned_rows = group[group["attack_type"] != "none"]
        for _, poisoned_row in poisoned_rows.iterrows():
            poisoned_acts = extractor.extract_for_document(poisoned_row["content"])
 
            for layer_idx in RAG_LAYERS:
                diff = poisoned_acts[layer_idx] - clean_acts[layer_idx]
                all_rows.append({
                    "pair_id": pair_id,
                    "attack_type": poisoned_row["attack_type"],
                    "layer_idx": layer_idx,
                    "diff_activation": diff,
                })
 
        if (i + 1) % 50 == 0:
            print(f"processed {i + 1}/{len(grouped)} pairs")
 
    torch.save(all_rows, OUTPUT_PATH)
    print(f"saved {len(all_rows)} rows to {OUTPUT_PATH}")
 
 
if __name__ == "__main__":
    main()
    
