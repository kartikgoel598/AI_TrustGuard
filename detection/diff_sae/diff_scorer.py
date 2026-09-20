import os
import torch
 
from detection.base import Detector, LayerResult, ValidationError
from detection.diff_sae.model import DiffSAE
 
MODEL_DIR = "detection/diff_sae/trained_models_pooled"
 
FEATURE_CONFIG = [
    {"layer": 14, "feature_idx": 161, "threshold": None},
    {"layer": 26, "feature_idx": 651, "threshold": None},
]
 
 
ARCH_THRESHOLDS = {
    "lora_r8": {14: 1.6069, 26: 0.1646},
    "lora_r32": {14: 2.1907, 26: 0.3424},
    "full-rank": {14: 0.2421, 26: 3.7630},
}
 
CONSERVATIVE_THRESHOLDS = {14: 2.1907, 26: 3.7630}
 
 
CONSERVATIVE_THRESHOLDS = {14: 2.1907, 26: 3.7630}
 
 
class DiffSAEScorer(Detector):
    def __init__(self, device="cpu"):
        self.device = device
        self.models = {}
        self.scale_factors = {}
 
        for cfg in FEATURE_CONFIG:
            layer = cfg["layer"]
            checkpoint = torch.load(os.path.join(MODEL_DIR, f"diff_sae_layer{layer}.pt"))
            model = DiffSAE(input_dim=960, expansion_factor=4).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
 
            self.models[layer] = model
            self.scale_factors[layer] = checkpoint["scale_factor"]
 
    def set_thresholds(self, thresholds_by_layer):
        global CONSERVATIVE_THRESHOLDS
        CONSERVATIVE_THRESHOLDS = thresholds_by_layer
 
    def validate(self, input_data):
        for cfg in FEATURE_CONFIG:
            layer_key = f"layer_{cfg['layer']}"
            if layer_key not in input_data:
                return ValidationError(layer="layer1_diff_sae", message=f"missing {layer_key}", field=layer_key)
 
            tensor = input_data[layer_key]
            if not isinstance(tensor, torch.Tensor):
                return ValidationError(layer="layer1_diff_sae", message=f"{layer_key} is not a tensor", field=layer_key)
 
            if tensor.shape[-1] != 960:
                return ValidationError(layer="layer1_diff_sae", message=f"{layer_key} has wrong shape {tensor.shape}", field=layer_key)
 
        return None
 
    def _get_feature_activations(self, input_data):
        activations = {}
        for cfg in FEATURE_CONFIG:
            layer = cfg["layer"]
            feat_idx = cfg["feature_idx"]
 
            raw_vec = input_data[f"layer_{layer}"].float()
            scaled_vec = raw_vec * self.scale_factors[layer]
 
            with torch.no_grad():
                features = self.models[layer].encode(scaled_vec.unsqueeze(0).to(self.device))
 
            activations[layer] = features[0, feat_idx].item()
 
        return activations
 
    def _check_against_thresholds(self, activations, thresholds):
        flagged_any = False
        detail = {}
 
        for cfg in FEATURE_CONFIG:
            layer = cfg["layer"]
            feat_idx = cfg["feature_idx"]
            activation_val = activations[layer]
            threshold = thresholds[layer]
            fired = activation_val > threshold
 
            detail[f"layer_{layer}_feature_{feat_idx}"] = {
                "activation": activation_val,
                "threshold": threshold,
                "fired": fired,
            }
 
            if fired:
                flagged_any = True
 
        return flagged_any, detail
 
    def run(self, input_data):
        error = self.validate(input_data)
        if error is not None:
            return LayerResult(risk_score=0.0, status="error", detail={"error": error.message})
 
        activations = self._get_feature_activations(input_data)
 
        declared_arch = input_data.get("declared_architecture", "unknown")
 
        if declared_arch == "lora":
            flagged, detail = self._check_against_thresholds(activations, ARCH_THRESHOLDS["lora_r8"])
            status = "flagged" if flagged else "clean"
            return LayerResult(
                risk_score=1.0 if flagged else 0.0,
                status=status,
                detail={"mode": "declared_lora", "result": detail},
            )
 
        if declared_arch == "full-rank":
            flagged, detail = self._check_against_thresholds(activations, ARCH_THRESHOLDS["full-rank"])
            status = "flagged" if flagged else "clean"
            return LayerResult(
                risk_score=1.0 if flagged else 0.0,
                status=status,
                detail={"mode": "declared_full-rank", "result": detail},
            )
 
        if declared_arch == "qlora":
            flagged, detail = self._check_against_thresholds(activations, ARCH_THRESHOLDS["lora_r8"])
            status = "flagged" if flagged else "clean"
            return LayerResult(
                risk_score=1.0 if flagged else 0.0,
                status=status,
                detail={"mode": "declared_qlora_using_lora_thresholds_unvalidated", "result": detail},
            )
 
        per_arch_results = {}
        for arch_name, thresholds in ARCH_THRESHOLDS.items():
            flagged, detail = self._check_against_thresholds(activations, thresholds)
            per_arch_results[arch_name] = {"flagged": flagged, "detail": detail}
 
        conservative_flagged, conservative_detail = self._check_against_thresholds(activations, CONSERVATIVE_THRESHOLDS)
 
        status = "flagged" if conservative_flagged else "clean"
        risk_score = 1.0 if conservative_flagged else 0.0
 
        return LayerResult(
            risk_score=risk_score,
            status=status,
            detail={
                "mode": "unknown_architecture_conservative",
                "conservative_result": conservative_detail,
                "per_architecture_breakdown": per_arch_results,
            },
        )