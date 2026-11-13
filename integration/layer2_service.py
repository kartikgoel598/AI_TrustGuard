from pathlib import Path


def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_layer2_test(uploaded_files):
    if not uploaded_files:
        raise ValueError("Please upload at least one .txt document.")

    documents = []
    names = []
    for f in uploaded_files:
        if Path(f.name).suffix.lower() != ".txt":
            raise ValueError(f"'{f.name}' is not a .txt file.")
        text = f.getvalue().decode("utf-8", errors="ignore").strip()
        if text:
            documents.append(text)
            names.append(f.name)

    if not documents:
        raise ValueError("All uploaded files were empty.")

    from detection.rag_poison_detector.product.pipeline import run_layer2

    result = run_layer2(documents, names, device=get_device())

    if result.status == "error":
        raise ValueError(result.detail.get("error", "Layer 2 failed."))

    detail = result.detail
    high = detail["high_confidence_files"]
    review = detail["review_files"]

    if high:
        status = "Poison Found"
        message = f"{len(high)} file(s) flagged by both detectors, and {len(review)} more need review."
    elif review:
        status = "Needs Review"
        message = f"No high-confidence poison, but {len(review)} file(s) were flagged by one detector."
    else:
        status = "Clean"
        message = "No file was flagged by either detector."

    return {
        "layer": "Layer 2",
        "status": status,
        "risk_score": result.risk_score,
        "backend_connected": True,
        "message": message,
        "n_documents": detail["n_documents"],
        "high_confidence_files": high,
        "review_files": review,
        "warnings": detail["warnings"],
    }