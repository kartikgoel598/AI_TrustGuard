import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold

runs = torch.load("fake_runs.pt", weights_only=False)
layers = [12, 18, 24, 30]

labels = np.array([r["label"] for r in runs])
groups = np.array([r["scenario_id"][:3] for r in runs])
types = np.array([r["run_type"] for r in runs])

print(f"{len(runs)} runs | attacked: {labels.sum()} | normal: {(labels == 0).sum()}")


def get_features(step_index, layer):
    return np.stack([r["steps"][step_index]["hidden"][layer].numpy() for r in runs])


def test_probe(X):
    predictions = np.zeros(len(labels), dtype=int)
    folds = GroupKFold(n_splits=5)
    for train_idx, test_idx in folds.split(X, labels, groups):
        probe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        probe.fit(X[train_idx], labels[train_idx])
        predictions[test_idx] = probe.predict(X[test_idx])
    return predictions


def show(name, predictions):
    accuracy = (predictions == labels).mean()
    caught = {t: predictions[types == t].mean() for t in ["A", "B", "D"]}
    print(f"{name:<22} accuracy={accuracy:.2f} | flagged A={caught['A']:.2f} "
          f"B={caught['B']:.2f} D={caught['D']:.2f}")


print("\nResult moment (step 1) - main test")
for layer in layers:
    show(f"  layer {layer}", test_probe(get_features(1, layer)))

print("\nPlan moment (step 0) - cheat check, should be near guessing")
for layer in layers:
    show(f"  layer {layer}", test_probe(get_features(0, layer)))

print("\nText checker baseline")
baseline = []
for r in runs:
    result_text = str(r["steps"][0]["tool_result"])
    baseline.append(0 if r["expected"]["destination"] in result_text else 1)
show("  destination match", np.array(baseline))