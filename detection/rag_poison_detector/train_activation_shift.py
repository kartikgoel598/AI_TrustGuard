import os
import torch
from torch.utils.data import Dataset, DataLoader

from detection.diff_sae.model import DiffSAE, compute_loss

DIFF_DATA_PATH = "activation_capture/rag_diff/rag_diff_data.pt"
OUTPUT_DIR = "detection/rag_poison_detector/trained_models"

LAYERS = [14, 26]
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
SPARSITY_LAMBDA = 0.5
NUM_EPOCHS = 20


class DiffActivationDataset(Dataset):
    def __init__(self, activations):
        self.activations = activations

    def __len__(self):
        return self.activations.shape[0]

    def __getitem__(self, idx):
        return self.activations[idx]


def load_layer_data(layer_idx):
    rows = torch.load(DIFF_DATA_PATH)

    vectors = []
    for row in rows:
        if row["layer_idx"] == layer_idx:
            vectors.append(row["diff_activation"])

    stacked = torch.stack(vectors).float()

    avg_squared_norm = (stacked ** 2).sum(dim=1).mean()
    scale_factor = (stacked.shape[1] / avg_squared_norm).sqrt()
    stacked = stacked * scale_factor

    return stacked, scale_factor


def measure_dead_features(model, activations, device, batch_size=512):
    model.eval()
    n_features = model.n_features
    ever_active = torch.zeros(n_features, dtype=torch.bool)

    with torch.no_grad():
        for i in range(0, activations.shape[0], batch_size):
            batch = activations[i:i + batch_size].to(device)
            features = model.encode(batch)
            ever_active |= (features > 0).any(dim=0).cpu()

    model.train()
    return (~ever_active).float().mean().item()


def train_one_layer(layer_idx, device):
    print(f"\n{'=' * 60}")
    print(f"Training RAG ActivationShift SAE for layer {layer_idx}")
    print(f"{'=' * 60}")

    activations, scale_factor = load_layer_data(layer_idx)
    print(f"loaded {activations.shape[0]} diff-activation vectors")
    print(f"scale factor: {scale_factor:.6f}")

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
                batch, reconstruction, features, sparsity_lambda=SPARSITY_LAMBDA
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

        print(f"epoch {epoch + 1:3d}/{NUM_EPOCHS} | loss={avg_loss:.6f} | recon={avg_recon:.6f} | "
              f"sparsity_loss={avg_sparsity:.6f} | pct_zero(snapshot)={pct_zero_snapshot:.2%}")

    dead_frac = measure_dead_features(model, activations, device)
    print(f"dead feature fraction (full dataset scan): {dead_frac:.2%}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"activation_shift_sae_layer{layer_idx}.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "scale_factor": scale_factor,
        "layer_idx": layer_idx,
    }, out_path)
    print(f"saved -> {out_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device: {device}")

    for layer_idx in LAYERS:
        train_one_layer(layer_idx, device)

    print("\nall layers trained.")


if __name__ == "__main__":
    main()