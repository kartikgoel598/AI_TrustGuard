import torch
import torch.nn as nn


class DiffSAE(nn.Module):
    def __init__(self, input_dim: int = 960, expansion_factor: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.n_features = input_dim * expansion_factor  # 960 * 4 = 3840

        self.encoder = nn.Linear(input_dim, self.n_features)
        self.decoder = nn.Linear(self.n_features, input_dim)
        with torch.no_grad():
            self.decoder.weight.data = torch.nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )
            self.encoder.weight.data = self.decoder.weight.data.t().clone()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, input_dim] -> features: [batch, n_features], sparse (mostly zero)."""
        return torch.relu(self.encoder(x))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """features: [batch, n_features] -> reconstruction: [batch, input_dim]."""
        return self.decoder(features)

    def forward(self, x: torch.Tensor):
        features = self.encode(x)
        reconstruction = self.decode(features)
        return reconstruction, features

    def renormalize_decoder(self):
        with torch.no_grad():
            self.decoder.weight.data = torch.nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )


def compute_loss(x: torch.Tensor, reconstruction: torch.Tensor, features: torch.Tensor, sparsity_lambda: float = 1e-4):
    reconstruction_loss = torch.nn.functional.mse_loss(reconstruction, x)
    sparsity_loss = features.abs().mean()
    total_loss = reconstruction_loss + sparsity_lambda * sparsity_loss
    return total_loss, reconstruction_loss, sparsity_loss