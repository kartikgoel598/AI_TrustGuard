
import argparse

import torch

from detection.diff_sae.data import (ARCHS, LAYERS, fmt, get_device, load_features, metrics,
                                     or_flags, per_arch_recall, rank_features)

TOP_K = 5


def print_top(top):
    for r in top:
        print(f"  feature #{r['feature_idx']:5d} | {fmt(r)}")


def score_heldout(device):
    per_layer, or_metrics, all_flags, ref_labels = {}, {}, None, None
    for layer in LAYERS:
        print(f"\n{'=' * 60}\nLayer {layer} on HELD-OUT (full-rank) data\n{'=' * 60}")
        feats, is_trig, _, labels = load_features(layer, "heldout", ["full-rank"], device)
        print(f"{len(labels)} rows ({labels.count('clean')} clean, {labels.count('triggered')} triggered)")

        top = rank_features(feats, is_trig, TOP_K)
        print_top(top)
        flags = or_flags(feats, top)
        or_metrics[layer] = metrics(flags, is_trig)
        per_layer[layer] = top
        print(f"OR-logic top-{TOP_K}: {fmt(or_metrics[layer])}")

      
        if ref_labels is None:
            ref_labels, all_flags = labels, flags
        elif labels != ref_labels:
            raise ValueError(f"Rows not aligned between layer {LAYERS[0]} and {layer}; cross-layer OR invalid.")
        else:
            all_flags = all_flags | flags

    torch.save({"per_feature": per_layer, "or_logic": or_metrics}, "detection/diff_sae/bis_scores.pt")
    print(f"\n{'=' * 60}\nSUMMARY (OR-logic top-{TOP_K} per layer)")
    for layer, m in or_metrics.items():
        print(f"  Layer {layer}: {fmt(m)}")

    cross = metrics(all_flags, torch.tensor([l == "triggered" for l in ref_labels]))
    print(f"\nCROSS-LAYER (all {len(LAYERS)} layers, top-{TOP_K} each, OR): {fmt(cross)}")
    print(f"  TP={cross['tp']} FN={cross['fn']} FP={cross['fp']} TN={cross['tn']}")
    torch.save({k: cross[k] for k in ("bis", "precision", "recall", "fpr", "tp", "fn", "fp", "tn")},
               "detection/diff_sae/cross_layer_result.pt")


def score_diagnostic(device):
    for layer in LAYERS:
        print(f"\n[DIAGNOSTIC] Layer {layer} on TRAINING data (LoRA r8+r32, NOT held out)")
        feats, is_trig, _, labels = load_features(layer, "heldout", ["lora_r8", "lora_r32"], device)
        print(f"  {len(labels)} rows ({labels.count('clean')} clean, {labels.count('triggered')} triggered)")
        print_top(rank_features(feats, is_trig, TOP_K))


def score_pooled(device):
    summary = {}
    for layer in LAYERS:
        print(f"\n{'=' * 60}\nPOOLED layer {layer}\n{'=' * 60}")
        feats, is_trig, archs, labels = load_features(layer, "pooled", ARCHS, device)
        print(f"{len(labels)} rows ({labels.count('clean')} clean, {labels.count('triggered')} triggered)")
        top = rank_features(feats, is_trig, TOP_K)
        print_top(top)
        summary[layer] = metrics(or_flags(feats, top), is_trig)
        print(f"Combined top-{TOP_K}: {fmt(summary[layer])}")
        print(f"Per-architecture recall (feature #{top[0]['feature_idx']}):")
        per_arch_recall(feats[:, top[0]["feature_idx"]] > top[0]["threshold"], is_trig, archs)
    print(f"\n{'=' * 60}\nSUMMARY")
    for layer, m in summary.items():
        print(f"  Layer {layer}: {fmt(m)}")


def score_combined(device, features={14: 161, 26: 651}):
    flags, ref = None, None
    for layer, feat_idx in features.items():
        feats, is_trig, archs, labels = load_features(layer, "pooled", ARCHS, device)
        if ref is not None and (labels, archs) != ref:
            print("row order mismatch between layers, cannot combine safely")
            return
        ref = (labels, archs)
        vals = feats[:, feat_idx]
        threshold = torch.quantile(vals[~is_trig], 0.95).item()
        flags = (vals > threshold) if flags is None else flags | (vals > threshold)
    print(f"combined lora+fullrank feature detector\n{fmt(metrics(flags, is_trig))}")
    per_arch_recall(flags, is_trig, archs)


if __name__ == "__main__":
    modes = {"heldout": score_heldout, "diagnostic": score_diagnostic,
             "pooled": score_pooled, "combined": score_combined}
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=modes)
    modes[p.parse_args().mode](get_device())
