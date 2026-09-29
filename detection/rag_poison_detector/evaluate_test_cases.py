import pandas as pd
import torch

from model_loader import load_model_from_registry
from rag_extractor import RagActivationExtractor, RAG_LAYERS
from detection.diff_sae.model import DiffSAE
from detection.rag_poison_detector.score import score_layer, MODEL_DIR

CORPUS_PATH = "dataset/rag_dataset/rag_corpus.csv"
TEST_CASES_PATH = "dataset/rag_dataset/rag_test_cases.csv"
BASELINE_MODEL = "smollm2_360m_benign_full-rank"


def load_sae(layer_idx, device):
    checkpoint = torch.load(f"{MODEL_DIR}/activation_shift_sae_layer{layer_idx}.pt")
    model = DiffSAE(input_dim=960, expansion_factor=4).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["scale_factor"]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    corpus = pd.read_csv(CORPUS_PATH)
    contents = dict(zip(corpus["document_id"], corpus["content"]))
    tests = pd.read_csv(TEST_CASES_PATH)

    baseline = load_model_from_registry(BASELINE_MODEL)
    extractor = RagActivationExtractor(baseline["model"], baseline["tokenizer"], device)

    needed_ids = set()
    for ids in tests["corpus_document_ids"]:
        needed_ids.update(ids.split(";"))

    cache = {}
    for i, doc_id in enumerate(sorted(needed_ids)):
        cache[doc_id] = extractor.extract_for_document(contents[doc_id])
        if (i + 1) % 200 == 0:
            print(f"extracted {i + 1}/{len(needed_ids)} documents")

    detectors = {}
    for layer_idx in RAG_LAYERS:
        sae, scale = load_sae(layer_idx, device)
        top = score_layer(layer_idx, device)
        feat_idxs = [r["feature_idx"] for r in top]
        thresholds = torch.tensor([r["threshold"] for r in top])
        detectors[layer_idx] = (sae, scale, feat_idxs, thresholds)

    stats = {}

    for _, row in tests.iterrows():
        ids = row["corpus_document_ids"].split(";")
        targets = set() if pd.isna(row["target_poison_document_ids"]) else set(row["target_poison_document_ids"].split(";"))
        n = len(ids)

        for layer_idx in RAG_LAYERS:
            sae, scale, feat_idxs, thresholds = detectors[layer_idx]

            acts = torch.stack([cache[d][layer_idx] for d in ids]).float()
            baseline_vecs = (acts.sum(dim=0) - acts) / (n - 1)
            diffs = (acts - baseline_vecs) * scale

            with torch.no_grad():
                feats = sae.encode(diffs.to(device)).cpu()

            flagged = (feats[:, feat_idxs] > thresholds).any(dim=1).tolist()

            key = (row["test_type"], layer_idx)
            s = stats.setdefault(key, {"scenarios": 0, "caught": 0, "alarm": 0,
                                       "target_total": 0, "target_flagged": 0,
                                       "other_total": 0, "other_flagged": 0})
            s["scenarios"] += 1

            any_target_flagged = False
            any_flagged = any(flagged)
            for doc_id, was_flagged in zip(ids, flagged):
                if doc_id in targets:
                    s["target_total"] += 1
                    s["target_flagged"] += int(was_flagged)
                    any_target_flagged = any_target_flagged or was_flagged
                else:
                    s["other_total"] += 1
                    s["other_flagged"] += int(was_flagged)

            s["caught"] += int(any_target_flagged)
            s["alarm"] += int(any_flagged)

    print("\nresults per test_type and layer")
    for (test_type, layer_idx), s in sorted(stats.items()):
        target_rate = s["target_flagged"] / s["target_total"] if s["target_total"] > 0 else float("nan")
        other_rate = s["other_flagged"] / s["other_total"] if s["other_total"] > 0 else float("nan")
        print(f"{test_type:16s} layer {layer_idx}: scenarios={s['scenarios']} "
              f"poison docs flagged={target_rate:.3f} other docs flagged={other_rate:.3f} "
              f"scenario caught={s['caught'] / s['scenarios']:.3f} scenario alarm={s['alarm'] / s['scenarios']:.3f}")


if __name__ == "__main__":
    main()