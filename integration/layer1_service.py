from dataclasses import dataclass, asdict
from pathlib import Path
import tempfile

ALLOWED_EXTENSIONS = {".json"}
@dataclass
class Layer1Result:
    status: str
    risk_score: float | None
    diff_sae_score: float | None
    bis: float | None
    backend_connected: bool
    message: str

    def to_dict(self):
        return asdict(self)

def validate_activation_file(uploaded_file):
    if uploaded_file is None:
        raise ValueError("Please upload an activation file.")

    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(
            ALLOWED_EXTENSIONS = {".json"}
        )

    if uploaded_file.size == 0:
        raise ValueError("Uploaded file is empty.")

    return suffix

def save_uploaded_file(uploaded_file):
    suffix = validate_activation_file(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)

def run_real_backend(file_path, model_family):
    try:
        from detection.diff_sae.process_client_submission import process_client_submission
    except ImportError:
        return None

    return process_client_submission(str(file_path))

def run_layer1_test(uploaded_file, model_family):
    file_path = save_uploaded_file(uploaded_file)

    backend_result = run_real_backend(file_path, model_family)

    if backend_result is None:
        return Layer1Result(
            status="Backend Not Connected",
            risk_score=None,
            diff_sae_score=None,
            bis=None,
            backend_connected=False,
            message="Activation file validated successfully. Layer 1 backend is not connected yet."
        ).to_dict()

    return Layer1Result(
        status=backend_result.get("status", "Unknown"),
        risk_score=backend_result.get("risk_score"),
        diff_sae_score=backend_result.get("diff_sae_score"),
        bis=backend_result.get("bis"),
        backend_connected=True,
        message=backend_result.get(
            "message",
            "Layer 1 detection completed successfully."
        )
    ).to_dict()