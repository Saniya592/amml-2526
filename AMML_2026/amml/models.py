from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoencoder(nn.Module):
    """Exact architecture supplied in the official assignment ``src/model.py``."""

    def __init__(self):
        super().__init__()
        self.capacity = 64
        self.latent_dims = 20

        self.conv1 = nn.Conv2d(1, self.capacity, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(self.capacity, self.capacity * 2, kernel_size=4, stride=2, padding=1)
        self.fc_mu = nn.Linear(self.capacity * 2 * 7 * 7, self.latent_dims)
        self.fc_logvar = nn.Linear(self.capacity * 2 * 7 * 7, self.latent_dims)

        self.fc_decode = nn.Linear(self.latent_dims, self.capacity * 2 * 7 * 7)
        self.conv2_decode = nn.ConvTranspose2d(
            self.capacity * 2, self.capacity, kernel_size=4, stride=2, padding=1
        )
        self.conv1_decode = nn.ConvTranspose2d(
            self.capacity, 1, kernel_size=4, stride=2, padding=1
        )

    def encoder(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc_decode(z)
        x = x.view(x.size(0), self.capacity * 2, 7, 7)
        x = F.relu(self.conv2_decode(x))
        return torch.sigmoid(self.conv1_decode(x))

    @staticmethod
    def latent_sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encoder(x)
        z = self.latent_sample(mu, logvar)
        return self.decoder(z), mu, logvar


class MultiTaskVAEClassifier(nn.Module):
    """Task 2 model: supplied VAE plus a small classification head.

    The classifier consumes the deterministic posterior mean ``mu``. This
    preserves a stable representation for classification while the decoder is
    trained through the reparameterised latent sample.
    """

    def __init__(self, dropout: float = 0.20):
        super().__init__()
        self.vae = VariationalAutoencoder()
        self.classifier = nn.Sequential(
            nn.Linear(self.vae.latent_dims, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor):
        mu, logvar = self.vae.encoder(x)
        z = self.vae.latent_sample(mu, logvar)
        reconstruction = self.vae.decoder(z)
        logits = self.classifier(mu)
        return reconstruction, mu, logvar, logits

    def deterministic_forward(self, x: torch.Tensor):
        mu, logvar = self.vae.encoder(x)
        reconstruction = self.vae.decoder(mu)
        logits = self.classifier(mu)
        return reconstruction, mu, logvar, logits


def _torch_load_weights(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        # Compatibility fallback for checkpoints containing metadata.
        return torch.load(path, map_location="cpu", weights_only=False)


def _normalise_state_dict(state: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(state, nn.Module):
        state = state.state_dict()
    if isinstance(state, Mapping):
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            if key in state and isinstance(state[key], Mapping):
                state = state[key]
                break
    if not isinstance(state, Mapping):
        raise TypeError("Checkpoint does not contain a state dictionary.")

    normalised: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        clean_key = str(key)
        for prefix in ("module.", "model.", "vae."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
        normalised[clean_key] = value
    return normalised


def load_pretrained_vae(path: Path, device: torch.device) -> VariationalAutoencoder:
    if not path.exists():
        raise FileNotFoundError(path)
    model = VariationalAutoencoder()
    state = _normalise_state_dict(_torch_load_weights(path))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint {path.name} is incompatible with the supplied VAE. "
            f"Missing keys={missing}; unexpected keys={unexpected}"
        )
    model.to(device).eval()
    return model


def initialise_multitask_from_model0(
    model0_path: Path,
    *,
    device: torch.device,
    dropout: float,
) -> MultiTaskVAEClassifier:
    source = load_pretrained_vae(model0_path, device=torch.device("cpu"))
    model = MultiTaskVAEClassifier(dropout=dropout)
    model.vae.load_state_dict(source.state_dict(), strict=True)
    return model.to(device)
