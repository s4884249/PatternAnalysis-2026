"""PyTorch modules for the basic autoencoder and VQ-VAE teaching baseline.

This file intentionally depends only on PyTorch and the Python standard library.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """Three-stage convolutional encoder; spatial size is reduced by 8x."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32, latent_dim: int = 64):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8 if c1 % 8 == 0 else 1, c1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8 if c2 % 8 == 0 else 1, c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8 if c3 % 8 == 0 else 1, c3),
            nn.SiLU(inplace=True),
            nn.Conv2d(c3, latent_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class ConvDecoder(nn.Module):
    """Decoder paired with ConvEncoder; outputs intensities in [0, 1]."""

    def __init__(self, out_channels: int = 1, base_channels: int = 32, latent_dim: int = 64):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.network = nn.Sequential(
            nn.Conv2d(latent_dim, c3, kernel_size=3, stride=1, padding=1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8 if c2 % 8 == 0 else 1, c2),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(8 if c1 % 8 == 0 else 1, c1),
            nn.SiLU(inplace=True),
            nn.ConvTranspose2d(c1, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)


class BasicAutoencoder(nn.Module):
    """Deterministic convolutional autoencoder used as the reconstruction baseline."""

    model_type = "ae"

    def __init__(self, base_channels: int = 32, latent_dim: int = 64):
        super().__init__()
        self.encoder = ConvEncoder(base_channels=base_channels, latent_dim=latent_dim)
        self.decoder = ConvDecoder(base_channels=base_channels, latent_dim=latent_dim)
        self.latent_dim = latent_dim

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        latent = self.encode(images)
        reconstruction = self.decode(latent)
        return {"reconstruction": reconstruction, "latent": latent}


class GaussianAutoencoder(nn.Module):
    """VAE-style convolutional AE whose decoder is trained on sampled latent maps.

    The encoder predicts a diagonal Gaussian posterior q(z|x) at every latent
    spatial position. During training, z is sampled with the reparameterization
    trick and the KL term nudges q(z|x) toward an independent standard-normal
    prior. Generation can therefore draw z ~ N(0, I) rather than feeding the
    decoder arbitrary latent noise it never saw during training.

    ``encode`` returns the posterior mean for deterministic diagnostics and
    latent-feature comparisons. ``forward`` samples during training and uses
    the mean during evaluation so reconstruction metrics are reproducible.
    """

    model_type = "ae"
    latent_mode = "gaussian"

    def __init__(self, base_channels: int = 32, latent_dim: int = 64, kl_beta: float = 0.1):
        super().__init__()
        if kl_beta < 0:
            raise ValueError("kl_beta cannot be negative")
        # The encoder emits mean and log-variance maps concatenated by channel.
        self.encoder = ConvEncoder(base_channels=base_channels, latent_dim=2 * latent_dim)
        self.decoder = ConvDecoder(base_channels=base_channels, latent_dim=latent_dim)
        self.latent_dim = int(latent_dim)
        self.kl_beta = float(kl_beta)

    def encode_distribution(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        parameters = self.encoder(images)
        mu, logvar = parameters.chunk(2, dim=1)
        # Prevent exp(logvar) overflow while retaining a broad useful range.
        return mu, logvar.clamp(min=-10.0, max=8.0)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode_distribution(images)
        return mu

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode_distribution(images)
        latent = self.reparameterize(mu, logvar) if self.training else mu
        reconstruction = self.decode(latent)
        # Mean KL per latent scalar keeps the loss scale comparable across image
        # sizes; kl_beta controls the reconstruction/regularization trade-off.
        kl_loss = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).mean()
        return {
            "reconstruction": reconstruction,
            "latent": latent,
            "mu": mu,
            "logvar": logvar,
            "kl_loss": kl_loss,
        }


class VectorQuantizer(nn.Module):
    """Nearest-neighbour vector quantizer with a straight-through estimator."""

    def __init__(self, num_embeddings: int = 256, embedding_dim: int = 64, commitment_beta: float = 0.25):
        super().__init__()
        if num_embeddings < 2:
            raise ValueError("num_embeddings must be at least 2")
        if commitment_beta < 0:
            raise ValueError("commitment_beta cannot be negative")
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_beta = float(commitment_beta)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def forward(self, z_e: torch.Tensor):
        """Return straight-through latents, codebook loss, commitment loss, IDs, perplexity."""
        if z_e.ndim != 4 or z_e.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected [batch, {self.embedding_dim}, height, width], got {tuple(z_e.shape)}"
            )
        # Convert BCHW -> (B*H*W, D) for vector-distance computation.
        vectors = z_e.permute(0, 2, 3, 1).contiguous()
        flat = vectors.view(-1, self.embedding_dim)
        codebook = self.embedding.weight
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
            - 2.0 * flat @ codebook.t()
        )
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view_as(vectors).permute(0, 3, 1, 2).contiguous()

        codebook_loss = F.mse_loss(quantized, z_e.detach())
        commitment_loss = self.commitment_beta * F.mse_loss(z_e, quantized.detach())
        # Decoder gradients flow to the encoder; codebook gradients come from codebook_loss.
        quantized_st = z_e + (quantized - z_e).detach()

        counts = torch.bincount(indices, minlength=self.num_embeddings).to(z_e.dtype)
        probabilities = counts / counts.sum().clamp_min(1.0)
        nonzero = probabilities > 0
        perplexity = torch.exp(-(probabilities[nonzero] * probabilities[nonzero].log()).sum())
        ids = indices.view(z_e.shape[0], z_e.shape[2], z_e.shape[3])
        active_codes = nonzero.sum().to(z_e.dtype)
        return quantized_st, codebook_loss, commitment_loss, ids, perplexity, active_codes


class VQVAE(nn.Module):
    """Convolutional VQ-VAE; samples are produced using a learned code-index prior."""

    model_type = "vqvae"

    def __init__(
        self,
        base_channels: int = 32,
        latent_dim: int = 64,
        codebook_size: int = 256,
        commitment_beta: float = 0.25,
    ):
        super().__init__()
        self.encoder = ConvEncoder(base_channels=base_channels, latent_dim=latent_dim)
        self.quantizer = VectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=latent_dim,
            commitment_beta=commitment_beta,
        )
        self.decoder = ConvDecoder(base_channels=base_channels, latent_dim=latent_dim)
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return continuous encoder output before quantization."""
        return self.encoder(images)

    def quantize(self, latent: torch.Tensor):
        return self.quantizer(latent)

    def encode_indices(self, images: torch.Tensor) -> torch.Tensor:
        z_e = self.encode(images)
        # The index map is all the autoregressive code prior needs.
        return self.quantizer(z_e)[3]

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode a [B,H,W] integer code-index grid."""
        quantized = self.quantizer.embedding(indices).permute(0, 3, 1, 2).contiguous()
        return self.decode(quantized)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_e = self.encode(images)
        z_q_st, codebook_loss, commitment_loss, indices, perplexity, active_codes = self.quantizer(z_e)
        reconstruction = self.decode(z_q_st)
        return {
            "reconstruction": reconstruction,
            "latent": z_e,
            "quantized": z_q_st,
            "indices": indices,
            "codebook_loss": codebook_loss,
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
            "active_codes": active_codes,
        }


class MaskedConv2d(nn.Conv2d):
    """2D convolution masked to expose only previous positions in raster order."""

    def __init__(self, mask_type: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if mask_type not in {"A", "B"}:
            raise ValueError("mask_type must be 'A' or 'B'")
        self.mask_type = mask_type
        mask = torch.ones_like(self.weight)
        kh, kw = self.kernel_size
        cy, cx = kh // 2, kw // 2
        mask[:, :, cy + 1 :, :] = 0
        mask[:, :, cy, cx + 1 :] = 0
        if mask_type == "A":
            mask[:, :, cy, cx] = 0
        self.register_buffer("mask", mask)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            inputs,
            self.weight * self.mask,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class CodePrior(nn.Module):
    """Small PixelCNN-like prior over discrete VQ-VAE code indices.

    Logits at location (row, col) depend only on earlier code indices under
    row-major raster order. Sampling is sequential and therefore slower than a
    parallel decoder, but is a defensible, explicit prior for this prototype.
    """

    def __init__(self, codebook_size: int = 256, hidden_channels: int = 64):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.hidden_channels = int(hidden_channels)
        self.embedding = nn.Embedding(self.codebook_size, hidden_channels)
        self.input_conv = MaskedConv2d(
            "A", hidden_channels, hidden_channels, kernel_size=7, padding=3
        )
        self.hidden_convs = nn.ModuleList(
            [MaskedConv2d("B", hidden_channels, hidden_channels, kernel_size=3, padding=1) for _ in range(4)]
        )
        self.output = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, self.codebook_size, kernel_size=1),
        )

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 3:
            raise ValueError(f"Expected code map [B,H,W], got {tuple(indices.shape)}")
        x = self.embedding(indices.long()).permute(0, 3, 1, 2).contiguous()
        x = F.relu(self.input_conv(x))
        for layer in self.hidden_convs:
            x = x + F.relu(layer(x))
        return self.output(x)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        height: int,
        width: int,
        device: torch.device,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sequentially sample [N,H,W] code-index grids from the learned prior."""
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        tokens = torch.zeros((num_samples, height, width), dtype=torch.long, device=device)
        for row in range(height):
            for col in range(width):
                logits = self(tokens)[:, :, row, col] / temperature
                probs = torch.softmax(logits, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
                tokens[:, row, col] = sampled
        return tokens


def build_model(model_type: str, config: Dict) -> nn.Module:
    """Instantiate a model from a checkpoint/config dictionary."""
    shared = {
        "base_channels": int(config.get("base_channels", 32)),
        "latent_dim": int(config.get("latent_dim", 64)),
    }
    if model_type == "ae":
        # Checkpoints made before Gaussian AE support omit this field and remain
        # loadable as deterministic AEs. New training defaults to Gaussian mode.
        if config.get("ae_latent_mode", "deterministic") == "gaussian":
            return GaussianAutoencoder(
                **shared,
                kl_beta=float(config.get("ae_kl_beta", 0.1)),
            )
        return BasicAutoencoder(**shared)
    if model_type == "vqvae":
        return VQVAE(
            **shared,
            codebook_size=int(config.get("codebook_size", 256)),
            commitment_beta=float(config.get("commitment_beta", 0.25)),
        )
    raise ValueError(f"Unknown model_type={model_type!r}; use 'ae' or 'vqvae'.")


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
