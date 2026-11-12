
from detection.base import LayerResult
from detection.diff_sae.diff_scorer import FEATURES


def get_fired_layers(diff_result):
    d = diff_result.detail
    per_feature = (d.get("conservative_result", {}) if d.get("mode") == "unknown_architecture_conservative"
                   else d.get("result", {}))
    return [layer for layer, feat in FEATURES.items()
            if per_feature.get(f"layer_{layer}_feature_{feat}", {}).get("fired")]


def combine_layer1_results(diff_result, matcher_result):
    diff_flagged = diff_result.status == "flagged"
    matcher_flagged = matcher_result.status == "flagged"
    fired_layers = get_fired_layers(diff_result)
    layer_text = ", ".join(map(str, fired_layers)) or "unspecified"
    matched = (f"{matcher_result.detail.get('matched_count', 0)}/"
               f"{matcher_result.detail.get('total_texts', 0)}")

    if matcher_flagged:
        outcome = "Confirmed Backdoor"
        explanation = (
            f"Anomalous activation detected at layer(s) {layer_text}, and output matched the "
            f"known backdoor payload on {matched} probes." if diff_flagged else
            f"Output matched the known backdoor payload on {matched} probes, even though activation "
            f"analysis did not flag it. Direct output evidence is treated as sufficient.")
    elif diff_flagged:
        outcome = "Manual Review"
        explanation = (f"Anomalous activation detected at layer(s) {layer_text}, but output did not match "
                       f"any known backdoor payload pattern. Could be a novel/untested trigger or a false positive.")
    else:
        outcome = "Benign"
        explanation = "No anomalous activation pattern and no output match to known backdoor payload."

    return LayerResult(
        risk_score=max(diff_result.risk_score, matcher_result.risk_score),
        status=outcome,
        detail={"explanation": explanation, "fired_layers": fired_layers,
                "diff_sae_status": diff_result.status, "trigger_pattern_status": matcher_result.status,
                "diff_sae_detail": diff_result.detail, "trigger_pattern_detail": matcher_result.detail},
    )
