
import argparse
import os

import torch
from torch.utils.data import DataLoader

from detection.diff_sae.data import (ARCHS, LAYERS, MODEL_DIRS, compute_scale_factor,
                                     get_device, load_diff_rows)
from detection.diff_sae.model import DiffSAE, compute_loss

BATCH_SIZE = 256
LEARNING_RATE = 1e-4
SPARSITY_LAMBDA = 0.5
NUM_EPOCHS = 20


SETUPS = {
    "heldout": (["lora_r8", "lora_r32"], "full-rank"),
    "pooled": (ARCHS, None),
}


def load_train_data(layer_idx, variant):
    x, _, _ = load_diff_rows(layer_idx, SETUPS[variant][0]) 
    scale = compute_scale_factor(x)
    return x * scale, scale


def train_sae(x, device, sparsity_lambda, num_epochs, verbose=True):
    loader = DataLoader(x, batch_size=BATCH_SIZE, shuffle=True)  
    model = DiffSAE(input_dim=x.shape[1], expansion_factor=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(num_epochs):
        totals, n = torch.zeros(3), 0
        for batch in loader:
            batch = batch.to(device)
            reconstruction, features = model(batch)
            loss, recon, sparsity = compute_loss(batch, reconstruction, features, sparsity_lambda)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.renormalize_decoder()  
            totals += torch.tensor([loss.item(), recon.item(), sparsity.item()])
            n += 1
        if verbose:
            avg = totals / n
            pct_zero = (features == 0).float().mean().item()  # last batch only
            print(f"Epoch {epoch + 1:3d}/{num_epochs} | loss={avg[0]:.6f} | recon={avg[1]:.6f} | "
                  f"sparsity_loss={avg[2]:.6f} | pct_zero(snapshot)={pct_zero:.2%}")
    model.eval()
    return model


def measure(model, x, device, batch_size=512):
    
    recon_losses, ever_active, zero_frac = [], torch.zeros(model.n_features, dtype=torch.bool), []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            batch = x[i:i + batch_size].to(device)
            recon, feats = model(batch)
            recon_losses.append(torch.nn.functional.mse_loss(recon, batch, reduction="none").mean(dim=1))
            zero_frac.append((feats == 0).float().cpu())
            ever_active |= (feats > 0).any(dim=0).cpu()
    return (torch.cat(recon_losses).mean().item(), torch.cat(zero_frac).mean().item(),
            (~ever_active).float().mean().item())


def train_one_layer(layer_idx, variant, device):
    print(f"\n{'=' * 60}\nTraining Diff-SAE layer {layer_idx} ({variant}, lambda={SPARSITY_LAMBDA})\n{'=' * 60}")
    x, scale = load_train_data(layer_idx, variant)
    print(f"{x.shape[0]} vectors, scale factor {scale:.6f} (saved with model, never recomputed)")

    model = train_sae(x, device, SPARSITY_LAMBDA, NUM_EPOCHS)
    print(f"Dead feature fraction: {measure(model, x, device)[2]:.2%}")

    os.makedirs(MODEL_DIRS[variant], exist_ok=True)
    out_path = os.path.join(MODEL_DIRS[variant], f"diff_sae_layer{layer_idx}.pt")
    train_archs, held_out = SETUPS[variant]
    torch.save({"state_dict": model.state_dict(), "scale_factor": scale, "layer_idx": layer_idx,
                "train_arch_patterns": train_archs, "held_out_arch_pattern": held_out}, out_path)
    print(f"Saved -> {out_path}")


def sweep_lambda(layer_idx, variant, device, lambdas=(0.1, 0.3, 0.5, 1.0), num_epochs=15):
    x, _ = load_train_data(layer_idx, variant)
    for lam in lambdas:
        recon, pct_zero, dead = measure(train_sae(x, device, lam, num_epochs, verbose=False), x, device)
        print(f"lambda={lam:<8} recon_loss={recon:.6f}  avg_pct_zero={pct_zero:.2%}  dead_features={dead:.2%}")
    print("\nPick high avg_pct_zero, LOW dead_features, acceptable recon_loss (dead != sparse).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("variant", choices=SETUPS)
    p.add_argument("--sweep", action="store_true")
    args = p.parse_args()
    device = get_device()
    if args.sweep:
        sweep_lambda(14, args.variant, device)
    else:
        for layer in LAYERS:
            train_one_layer(layer, args.variant, device)
