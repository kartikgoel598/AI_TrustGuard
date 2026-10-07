import numpy as np
import torch

from detection.base import LayerResult
from model_loader import load_model_from_registry
from detection.rag_poison_detector.product.clean_base_detector import CleanBaseDetector, PRUNE_PERCENTILE
from detection.rag_poison_detector.product.activation_shift_detector import ActivationShiftDetector
from detection.rag_poison_detector.product.rag_extractor import RagActivationExtractor

BASELINE_MODEL = "smollm2_360m_benign_full-rank"

# from the collection size test: CleanBase recall was 0.25 at 20 documents and 0.6 or more from 50 up
MIN_CLEANBASE_DOCS = 50

BOTH = "high_confidence"
ONE = "needs_review"
NEITHER = "clean"
RANK = {NEITHER: 0, ONE: 1, BOTH: 2}


def extract_document_activations(documents, layers, device):
    baseline = load_model_from_registry(BASELINE_MODEL)
    extractor = RagActivationExtractor(baseline["model"], baseline["tokenizer"], device)

    per_layer = {layer: [] for layer in layers}
    for content in documents:
        acts = extractor.extract_for_document(content)
        for layer in layers:
            per_layer[layer].append(acts[layer])

    return {f"layer_{layer}": torch.stack(vecs).float().cpu() for layer, vecs in per_layer.items()}


def run_clean_base(documents, device):
    detector = CleanBaseDetector(device=device)
    result = detector.run({"documents": documents})

    flags = np.zeros(len(documents), dtype=bool)
    for item in result.detail["per_document"]:
        flags[item["index"]] = item["flagged"]

    return flags


def group_by_file(verdicts):
    files = {}
    for v in verdicts:
        entry = files.setdefault(v["file"], {"file": v["file"], "verdict": NEITHER, "flagged_chunks": 0, "total_chunks": 0})
        entry["total_chunks"] += 1
        if v["verdict"] != NEITHER:
            entry["flagged_chunks"] += 1
        if RANK[v["verdict"]] > RANK[entry["verdict"]]:
            entry["verdict"] = v["verdict"]

    return list(files.values())


def run_layer2(documents, file_names=None, device="cpu"):
    if len(documents) == 0:
        return LayerResult(risk_score=0.0, status="error", detail={"error": "no documents were provided"})

    if file_names is None:
        file_names = [f"document_{i}" for i in range(len(documents))]

    if len(file_names) != len(documents):
        return LayerResult(risk_score=0.0, status="error", detail={"error": "file_names must match documents one to one"})

    activation_detector = ActivationShiftDetector(device=device)

    warnings = []
    clean_base_active = len(documents) >= MIN_CLEANBASE_DOCS

    if clean_base_active:
        clean_base_flags = run_clean_base(documents, device)
    else:
        clean_base_flags = np.zeros(len(documents), dtype=bool)
        warnings.append(
            f"only {len(documents)} documents were uploaded, fewer than the {MIN_CLEANBASE_DOCS} needed for "
            f"clique detection to be reliable. Only the activation detector was used, so no file can be "
            f"marked high confidence and every flagged file needs manual review."
        )

    activations = extract_document_activations(documents, activation_detector.layers, device)
    error = activation_detector.validate(activations)
    if error is not None:
        return LayerResult(risk_score=0.0, status="error", detail={"error": error.message})
    activation_flags = activation_detector.flag_documents(activations)

    verdicts = []
    for name, a, b in zip(file_names, clean_base_flags, activation_flags):
        if a and b:
            verdict = BOTH
        elif a or b:
            verdict = ONE
        else:
            verdict = NEITHER

        verdicts.append({
            "file": name,
            "verdict": verdict,
            "clean_base_flagged": bool(a),
            "activation_shift_flagged": bool(b),
        })

    files = group_by_file(verdicts)
    high_confidence = sorted([f for f in files if f["verdict"] == BOTH], key=lambda f: -f["flagged_chunks"])
    review = sorted([f for f in files if f["verdict"] == ONE], key=lambda f: -f["flagged_chunks"])

    n_both_docs = sum(v["verdict"] == BOTH for v in verdicts)

    return LayerResult(
        risk_score=n_both_docs / len(documents),
        status="flagged" if (high_confidence or review) else "clean",
        detail={
            "n_documents": len(documents),
            "n_files": len(files),
            "high_confidence_files": [f["file"] for f in high_confidence],
            "review_files": [f["file"] for f in review],
            "files": files,
            "clean_base_active": clean_base_active,
            "min_clean_base_documents": MIN_CLEANBASE_DOCS,
            "prune_percentile": PRUNE_PERCENTILE,
            "warnings": warnings,
            "documents": verdicts,
        },
    )