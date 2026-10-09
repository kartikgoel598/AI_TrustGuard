import os
import torch

from detection.base import Detector, LayerResult, ValidationError
from detection.diff_sae.model import DiffSAE

MODEL_DIR = "detection/rag_poison_detector/trained_models"
CONFIG_PATH = os.path.join(MODEL_DIR, "activation_shift_config.pt")
HIDDEN_DIM = 960


def load_sae(layer_idx, device):
    checkpoint = torch.load(os.path.join(MODEL_DIR, f"raw_sae_layer{layer_idx}.pt"))
    model = DiffSAE(input_dim=HIDDEN_DIM, expansion_factor=4).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["mean"], checkpoint["scale"]


def encode(model, mean, scale, x, device):
    with torch.no_grad():
        return model.encode(((x - mean) * scale).to(device)).cpu()


def flag_with(feats_by_layer, picks):
    flagged = None
    for layer_idx, feat_idx, threshold in picks:
        current = feats_by_layer[layer_idx][:, feat_idx] > threshold
        flagged = current if flagged is None else (flagged | current)
    return flagged


class ActivationShiftDetector(Detector):
    def __init__(self, device="cpu"):
        self.device = device
        self.picks = torch.load(CONFIG_PATH)["picks"]
        self.layers = sorted(set(p[0] for p in self.picks))
        self.models = {layer: load_sae(layer, device) for layer in self.layers}

    def validate(self, input_data):
        for layer in self.layers:
            key = f"layer_{layer}"
            if key not in input_data:
                return ValidationError(layer="layer2_activation_shift", message=f"missing {key}", field=key)

            tensor = input_data[key]
            if not isinstance(tensor, torch.Tensor):
                return ValidationError(layer="layer2_activation_shift", message=f"{key} is not a tensor", field=key)

            if tensor.shape[-1] != HIDDEN_DIM:
                return ValidationError(layer="layer2_activation_shift", message=f"{key} has wrong shape {tuple(tensor.shape)}", field=key)

        return None

    def flag_documents(self, input_data):
        features = {}
        for layer in self.layers:
            x = input_data[f"layer_{layer}"].float()
            if x.dim() == 1:
                x = x.unsqueeze(0)

            model, mean, scale = self.models[layer]
            features[layer] = encode(model, mean, scale, x, self.device)

        return flag_with(features, self.picks).numpy()

    def run(self, input_data):
        error = self.validate(input_data)
        if error is not None:
            return LayerResult(risk_score=0.0, status="error", detail={"error": error.message})

        flagged = self.flag_documents(input_data)

        return LayerResult(
            risk_score=float(flagged.mean()),
            status="flagged" if flagged.any() else "clean",
            detail={
                "flags": flagged.tolist(),
                "n_flagged": int(flagged.sum()),
                "n_documents": int(len(flagged)),
                "picks": [{"layer": l, "feature_idx": f, "threshold": t} for l, f, t in self.picks],
            },
        )