import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader

from detection.diff_sae.model import DiffSAE, compute_loss

DIFF_DATA_DIR = "activation_capture/diff_data"
OUTPUT_DIR = "detection/diff_sae/trained_models_pooled"

LAYERS = [14, 18, 22, 26]
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
SPARSITY_LAMBDA = 0.5
NUM_EPOCHS = 20

TRAIN_ARCH_PATTERNS = ["lora_r8", "lora_r32", "full-rank"]
HELD_OUT_ARCH_PATTERN = None


class DiffActivationDataset(Dataset):
    def __init__(self, activations: torch.Tensor):
        self.activations = activations

    def __len__(self):
        return self.activations.shape[0]

    def __getitem__(self, idx):
        return self.activations[idx]


def _load_and_normalize(diff_files: list, layer_idx: int, scale_factor: float = None):
    all_vectors = []
    for path in diff_files:
        rows = torch.load(path)
        for row in rows:
            if row["layer_idx"] == layer_idx:
                all_vectors.append(row["diff_activation"])

    if not all_vectors:
        raise ValueError(f"No rows found for layer {layer_idx} in {diff_files}.")

    stacked = torch.stack(all_vectors).float()

    if scale_factor is None:
        avg_squared_norm = (stacked ** 2).sum(dim=1).mean()
        scale_factor = (stacked.shape[1] / avg_squared_norm).sqrt()

    stacked = stacked * scale_factor
    return stacked, scale_factor


def load_train_data(layer_idx: int):
    all_files = glob.glob(os.path.join(DIFF_DATA_DIR, "*_diff.pt"))
    train_files = [f for f in all_files if any(p in f for p in TRAIN_ARCH_PATTERNS)]

    if not train_files:
        raise FileNotFoundError(
            f"No training files matched patterns {TRAIN_ARCH_PATTERNS} in {DIFF_DATA_DIR}."
        )

    print(f"  Training files: {[os.path.basename(f) for f in train_files]}")
    return _load_and_normalize(train_files, layer_idx, scale_factor=None)


def load_held_out_data(layer_idx: int, scale_factor: float):
    all_files = glob.glob(os.path.join(DIFF_DATA_DIR, "*_diff.pt"))
    held_out_files = [f for f in all_files if HELD_OUT_ARCH_PATTERN in f]

    if not held_out_files:
        raise FileNotFoundError(
            f"No held-out files matched pattern '{HELD_OUT_ARCH_PATTERN}' in {DIFF_DATA_DIR}."
        )

    print(f"  Held-out files: {[os.path.basename(f) for f in held_out_files]}")
    return _load_and_normalize(held_out_files, layer_idx, scale_factor=scale_factor)


def train_one_layer(layer_idx: int, device: str, sparsity_lambda: float = None):
    sparsity_lambda = sparsity_lambda if sparsity_lambda is not None else SPARSITY_LAMBDA

    print(f"\n{'=' * 60}")
    print(f"Training Diff-SAE for layer {layer_idx} (lambda={sparsity_lambda})")
    print(f"{'=' * 60}")

    activations, scale_factor = load_train_data(layer_idx)
    print(f"Loaded {activations.shape[0]} TRAINING diff-activation vectors "
          f"(architectures: {TRAIN_ARCH_PATTERNS}, clean+triggered pooled).")
    print(f"Scale factor: {scale_factor:.6f} (will be saved with model, "
          f"reused for held-out eval and future inference — never recomputed).")

    dataset = DiffActivationDataset(activations)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DiffSAE(input_dim=activations.shape[1], expansion_factor=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        total_loss, total_recon, total_sparsity = 0.0, 0.0, 0.0
        n_batches = 0

        for batch in dataloader:
            batch = batch.to(device)

            reconstruction, features = model(batch)
            loss, recon_loss, sparsity_loss = compute_loss(
                batch, reconstruction, features, sparsity_lambda=sparsity_lambda
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.renormalize_decoder()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_sparsity += sparsity_loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_sparsity = total_sparsity / n_batches

        with torch.no_grad():
            pct_zero_snapshot = (features == 0).float().mean().item()

        print(f"Epoch {epoch + 1:3d}/{NUM_EPOCHS} | "
              f"loss={avg_loss:.6f} | recon={avg_recon:.6f} | "
              f"sparsity_loss={avg_sparsity:.6f} | pct_zero(snapshot)={pct_zero_snapshot:.2%}")

    dead_frac = measure_dead_features(model, activations, device)
    print(f"Dead feature fraction (full dataset scan): {dead_frac:.2%}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"diff_sae_layer{layer_idx}.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "scale_factor": scale_factor,
        "layer_idx": layer_idx,
        "train_arch_patterns": TRAIN_ARCH_PATTERNS,
        "held_out_arch_pattern": HELD_OUT_ARCH_PATTERN,
    }, out_path)
    print(f"Saved trained Diff-SAE for layer {layer_idx} -> {out_path}")


def measure_dead_features(model, activations: torch.Tensor, device: str, batch_size: int = 512) -> float:
    model.eval()
    n_features = model.n_features
    ever_active = torch.zeros(n_features, dtype=torch.bool)

    with torch.no_grad():
        for i in range(0, activations.shape[0], batch_size):
            batch = activations[i:i + batch_size].to(device)
            features = model.encode(batch)
            ever_active |= (features > 0).any(dim=0).cpu()

    model.train()
    dead_fraction = (~ever_active).float().mean().item()
    return dead_fraction


def sweep_lambda_on_one_layer(layer_idx: int, device: str, lambda_values: list, num_epochs: int = 15):
    activations, scale_factor = load_train_data(layer_idx)
    print(f"Sweeping lambda on layer {layer_idx} ({activations.shape[0]} training vectors)\n")

    results = []
    for lam in lambda_values:
        dataset = DiffActivationDataset(activations)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        model = DiffSAE(input_dim=activations.shape[1], expansion_factor=4).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(num_epochs):
            for batch in dataloader:
                batch = batch.to(device)
                reconstruction, features = model(batch)
                loss, recon_loss, sparsity_loss = compute_loss(batch, reconstruction, features, sparsity_lambda=lam)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.renormalize_decoder()

        model.eval()
        with torch.no_grad():
            all_recon, all_features = [], []
            for i in range(0, activations.shape[0], 512):
                batch = activations[i:i + 512].to(device)
                recon, feats = model(batch)
                all_recon.append(torch.nn.functional.mse_loss(recon, batch, reduction='none').mean(dim=1))
                all_features.append(feats)
            full_recon_loss = torch.cat(all_recon).mean().item()
            full_features = torch.cat(all_features)
            pct_zero_avg = (full_features == 0).float().mean().item()

        dead_frac = measure_dead_features(model, activations, device)

        print(f"lambda={lam:<8} recon_loss={full_recon_loss:.6f}  "
              f"avg_pct_zero={pct_zero_avg:.2%}  dead_features={dead_frac:.2%}")
        results.append((lam, full_recon_loss, pct_zero_avg, dead_frac))

    print("\nPick the lambda with high avg_pct_zero, LOW dead_features (dead != sparse — "
          "a good result has high pct_zero but low dead_features), and acceptable recon_loss.")
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    for layer_idx in LAYERS:
        train_one_layer(layer_idx, device)

    print("\nAll layers trained.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sweep_lambda_on_one_layer(14, device, lambda_values=[0.1, 0.3, 0.5, 1.0], num_epochs=15)
    else:
        main()