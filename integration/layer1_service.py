from collections import Counter
from pathlib import Path
import os
import tempfile

ALLOWED_EXTENSIONS = {".json"}

CONFIRMED = "Confirmed Backdoor"
REVIEW = "Manual Review"
BENIGN = "Benign"


def validate_activation_file(uploaded_file):
    if uploaded_file is None:
        raise ValueError("Please upload an activation file.")
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{suffix}'. Please upload the .json file made by the extraction script.")
    if uploaded_file.size == 0:
        raise ValueError("Uploaded file is empty.")
    return suffix


def save_uploaded_file(uploaded_file):
    suffix = validate_activation_file(uploaded_file)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def run_real_backend(file_path, declared_architecture=None):
    try:
        from detection.diff_sae.process_client_submission import process_client_submission
    except ImportError as error:
        return None, str(error)
    return process_client_submission(str(file_path), declared_architecture=declared_architecture), None


def summarise(probes):
    counts = Counter(p.get("outcome", "Unknown") for p in probes)
    n = len(probes)

    if counts[CONFIRMED] > 0:
        status = CONFIRMED
        message = f"The backdoor payload appeared on {counts[CONFIRMED]} of {n} probes."
    elif counts[REVIEW] > 0:
        status = REVIEW
        message = f"Activation signals fired on {counts[REVIEW]} of {n} probes, but no payload appeared. A person should review this model."
    else:
        status = BENIGN
        message = f"No backdoor signal on any of the {n} probes."

    flagged = counts[CONFIRMED] + counts[REVIEW]
    risk_score = flagged / n if n else 0.0
    return status, risk_score, message, dict(counts)


def run_layer1_test(uploaded_file, declared_architecture=None):
    file_path = save_uploaded_file(uploaded_file)
    try:
        probes, import_error = run_real_backend(file_path, declared_architecture=declared_architecture)
    finally:
        try:
            os.remove(file_path)
        except OSError:
            pass

    if probes is None:
        return {
            "layer": "Layer 1",
            "status": "Backend Not Connected",
            "risk_score": None,
            "backend_connected": False,
            "message": f"The file was valid, but the Layer 1 backend could not be loaded ({import_error}).",
            "counts": {},
            "probes": [],
            "declared_architecture": None,
        }

    status, risk_score, message, counts = summarise(probes)
    architecture = probes[0].get("declared_architecture") if probes else None

    return {
        "layer": "Layer 1",
        "status": status,
        "risk_score": risk_score,
        "backend_connected": True,
        "message": message,
        "counts": counts,
        "probes": probes,
        "declared_architecture": architecture,
    }