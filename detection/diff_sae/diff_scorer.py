
import torch

from detection.base import Detector, LayerResult, ValidationError
from detection.diff_sae.data import HIDDEN_DIM, encode, load_sae

FEATURES = {14: 161, 26: 651}


ARCH_THRESHOLDS = {
    "lora_r8": {14: 1.6069, 26: 0.1646},
    "lora_r32": {14: 2.1907, 26: 0.3424},
    "full-rank": {14: 0.2421, 26: 3.7630},
}

CONSERVATIVE_THRESHOLDS = {14: 2.1907, 26: 3.7630}


DECLARED_MODES = {
    "lora": ("declared_lora", "lora_r8"),
    "full-rank": ("declared_full-rank", "full-rank"),
    "qlora": ("declared_qlora_using_lora_thresholds_unvalidated", "lora_r8"),
}


class DiffSAEScorer(Detector):
    def __init__(self, device="cpu", thresholds=None):
        self.device = device
        self.conservative = dict(thresholds or CONSERVATIVE_THRESHOLDS)
        self.saes = {layer: load_sae(layer, "pooled", device) for layer in FEATURES}  

    def set_thresholds(self, thresholds_by_layer):
        self.conservative = dict(thresholds_by_layer)

    def validate(self, input_data):
        for layer in FEATURES:
            key = f"layer_{layer}"
            tensor = input_data.get(key)
            if tensor is None:
                msg = f"missing {key}"
            elif not isinstance(tensor, torch.Tensor):
                msg = f"{key} is not a tensor"
            elif tensor.shape[-1] != HIDDEN_DIM:
                msg = f"{key} has wrong shape {tensor.shape}"
            else:
                continue
            return ValidationError(layer="layer1_diff_sae", message=msg, field=key)
        return None

    def _feature_values(self, input_data):
        """Scale each diff vector exactly like training data, encode, read the one feature we care about."""
        values = {}
        for layer, feat_idx in FEATURES.items():
            model, scale = self.saes[layer]
            x = (input_data[f"layer_{layer}"].float().to(self.device) * scale).unsqueeze(0)
            values[layer] = encode(model, x, self.device)[0, feat_idx].item()
        return values

    @staticmethod
    def _check(values, thresholds):
        """Flag if ANY feature is above its threshold."""
        detail = {f"layer_{layer}_feature_{feat}": {"activation": values[layer], "threshold": thresholds[layer],
                                                   "fired": values[layer] > thresholds[layer]}
                  for layer, feat in FEATURES.items()}
        return any(d["fired"] for d in detail.values()), detail

    def run(self, input_data):
        error = self.validate(input_data)
        if error is not None:
            return LayerResult(risk_score=0.0, status="error", detail={"error": error.message})
        values = self._feature_values(input_data)
        declared = input_data.get("declared_architecture", "unknown")
        

        if declared in DECLARED_MODES:
            mode, arch = DECLARED_MODES[declared]
            flagged, detail = self._check(values, ARCH_THRESHOLDS[arch])
            detail = {"mode": mode, "result": detail}

        else:
    
            flagged, conservative_detail = self._check(values, self.conservative)
            breakdown = {arch: dict(zip(("flagged", "detail"), self._check(values, t)))
                         for arch, t in ARCH_THRESHOLDS.items()}
            detail = {"mode": "unknown_architecture_conservative",
                      "conservative_result": conservative_detail,
                      "per_architecture_breakdown": breakdown}
            

        return LayerResult(risk_score=1.0 if flagged else 0.0,
                           status="flagged" if flagged else "clean", detail=detail)
