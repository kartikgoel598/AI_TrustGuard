from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "dataset/agent_dataset/dataset"
ACTS = ROOT / "activation_capture/agent"
MODEL_DIR = ROOT / "detection/agent_probe/trained_models"
LAYERS = [12, 18, 24, 30]
SEED = 42


def load_split(name):
    cases = pd.read_csv(DATA / f"agent_{name}_cases.csv").fillna("")
    scored = pd.read_csv(DATA / f"agent_{name}_attack_results.csv")
    hidden = torch.load(ACTS / f"agent_{name}_hidden_states.pt", weights_only=False)
    df = cases.merge(scored[["scenario_id", "misuse", "expected_divergence"]], on="scenario_id")
    df = df[df["scenario_id"].isin(hidden.keys())].reset_index(drop=True)
    df["group"] = df["scenario_type"]
    df.loc[(df.scenario_type == "attack") & (df.misuse == 1), "group"] = "attack_worked"
    df.loc[(df.scenario_type == "attack") & (df.misuse == 0), "group"] = "attack_resisted"
    df["prompt"] = df["user_request"] + " " + df["stated_plan"] + " " + df["adversarial_context"]
    return df, hidden


def features(df, hidden, layer):
    return np.stack([hidden[s][layer].numpy() for s in df["scenario_id"]])


def make_probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000, class_weight="balanced"))


def report(name, y_true, y_pred, groups):
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    neg = y_true == 0
    fpr = y_pred[neg].mean() if neg.any() else 0.0
    rates = " ".join(f"{g}={y_pred[groups == g].mean():.2f}" for g in sorted(set(groups)))
    print(f"{name:<18} precision={p:.2f} recall={r:.2f} false_alarm={fpr:.2f} | flagged: {rates}")


train, train_hidden = load_split("train")
test, test_hidden = load_split("test")
y_train = train["misuse"].to_numpy()
y_test = test["misuse"].to_numpy()

print(f"train: {len(train)} runs, misuse={y_train.sum()} | test: {len(test)} runs, misuse={y_test.sum()}")
print("train groups:", train["group"].value_counts().to_dict())
print("test groups: ", test["group"].value_counts().to_dict())

if y_train.sum() < 5 or y_test.sum() < 2:
    print("\nToo few misuse examples to train a probe. The agent resisted most attacks.")
    raise SystemExit

print("\nProbe on hidden states (about-to-act moment)")
best_layer, best_score = None, (-1, -1)
for layer in LAYERS:
    probe = make_probe().fit(features(train, train_hidden, layer), y_train)
    pred = probe.predict(features(test, test_hidden, layer))
    report(f"  layer {layer}", y_test, pred, test["group"].to_numpy())
    score = (recall_score(y_test, pred, zero_division=0), precision_score(y_test, pred, zero_division=0))
    if score > best_score:
        best_layer, best_score = layer, score

print("\nText baseline (prompt words only, no brain)")
text_model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=5000, class_weight="balanced"))
text_model.fit(train["prompt"], y_train)
report("  tf-idf", y_test, text_model.predict(test["prompt"]), test["group"].to_numpy())

print("\nCheat check (shuffled labels, should be near guessing)")
rng = np.random.default_rng(SEED)
shuffled = rng.permutation(y_train)
probe = make_probe().fit(features(train, train_hidden, best_layer), shuffled)
report(f"  layer {best_layer}", y_test, probe.predict(features(test, test_hidden, best_layer)), test["group"].to_numpy())

final = make_probe().fit(features(train, train_hidden, best_layer), y_train)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump({"probe": final, "layer": best_layer, "layers_captured": LAYERS}, MODEL_DIR / "layer3_probe.joblib")
print(f"\nSaved probe (layer {best_layer}) to {MODEL_DIR / 'layer3_probe.joblib'}")