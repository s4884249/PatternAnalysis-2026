"""PyTorch modules: a convolutional VAE and a VQ-VAE with a PixelCNN-style code prior.

Both models share the same conv encoder/decoder stack so differences come from the latent space
(VAE: one global Gaussian vector; VQ-VAE: a grid of discrete codes).
Depends only on PyTorch and the standard library.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)


class ConvEncoder(nn.Module):
    """Convolutional encoder; each of `stages` stride-2 blocks halves the spatial size (8x for 3).

    GroupNorm (not BatchNorm) keeps behaviour independent of batch size and
    identical in train/eval mode; 8 groups when the channel count allows it.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32, latent_dim: int = 64, stages: int = 3):
        super().__init__()
        if stages not in (2, 3):
            raise ValueError("stages must be 2 (4x downsampling) or 3 (8x downsampling)")
        layers, previous = [], in_channels
        for i in range(stages):
            width = base_channels * 2**i
            layers += [nn.Conv2d(previous, width, kernel_size=4, stride=2, padding=1), _norm(width), nn.SiLU(inplace=True)]
            previous = width
        layers.append(nn.Conv2d(previous, latent_dim, kernel_size=3, stride=1, padding=1))
        self.network = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)


class ConvDecoder(nn.Module):
    """Decoder paired with ConvEncoder; outputs intensities in [0, 1]."""

    def __init__(self, out_channels: int = 1, base_channels: int = 32, latent_dim: int = 64, stages: int = 3):
        super().__init__()
        if stages not in (2, 3):
            raise ValueError("stages must be 2 (4x upsampling) or 3 (8x upsampling)")
        widths = [base_channels * 2**i for i in range(stages)]
        layers = [nn.Conv2d(latent_dim, widths[-1], kernel_size=3, stride=1, padding=1), nn.SiLU(inplace=True)]
        for i in range(stages - 1, 0, -1):
            layers += [
                nn.ConvTranspose2d(widths[i], widths[i - 1], kernel_size=4, stride=2, padding=1),
                _norm(widths[i - 1]),
                nn.SiLU(inplace=True),
            ]
        layers += [nn.ConvTranspose2d(widths[0], out_channels, kernel_size=4, stride=2, padding=1), nn.Sigmoid()]
        self.network = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.network(latent)


class VAE(nn.Module):
    """Convolutional VAE with ONE global Gaussian latent vector per image.

    Why a vector and not a feature map: generation decodes z ~ N(0, I). With a spatial
    latent (e.g. 64x16x10) every position is sampled independently, so nothing ties the
    positions together and the decoder receives latent maps it never saw in training
    (noise, no anatomy). A single global vector forces the whole image to be described
    by one code whose aggregate posterior the KL term pulls onto N(0, I), so prior
    samples land where the decoder has been trained.

    The encoder predicts mean and log-variance. Training samples z with the
    reparameterisation trick; evaluation uses the mean, which keeps reconstruction
    metrics reproducible.
    """

    model_type = "vae"
    feature_channels = 32  # channels of the conv feature map that is flattened into the vector

    def __init__(
        self,
        base_channels: int = 32,
        latent_dim: int = 256,
        kl_beta: float = 0.01,
        image_size: Tuple[int, int] = (128, 64),
        stages: int = 3,
    ):
        super().__init__()
        if kl_beta < 0:
            raise ValueError("kl_beta cannot be negative")
        height, width = int(image_size[0]), int(image_size[1])
        factor = 2**stages
        if height % factor or width % factor:
            raise ValueError(f"image_size must be divisible by {factor}")
        self.feature_shape = (self.feature_channels, height // factor, width // factor)
        flat = self.feature_channels * (height // factor) * (width // factor)
        self.encoder = ConvEncoder(base_channels=base_channels, latent_dim=self.feature_channels, stages=stages)
        self.to_posterior = nn.Linear(flat, 2 * latent_dim)  # mean and log-variance
        self.from_latent = nn.Linear(latent_dim, flat)
        self.decoder = ConvDecoder(base_channels=base_channels, latent_dim=self.feature_channels, stages=stages)
        self.latent_dim = int(latent_dim)
        self.kl_beta = float(kl_beta)
        self.num_pixels = height * width

    def encode_distribution(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.to_posterior(self.encoder(images).flatten(1)).chunk(2, dim=1)
        # Clamp so exp(logvar) cannot overflow while keeping a wide useful range.
        return mu, logvar.clamp(min=-10.0, max=8.0)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Posterior mean [B, latent_dim]: the deterministic latent used for diagnostics."""
        return self.encode_distribution(images)[0]

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.from_latent(latent).view(-1, *self.feature_shape))

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        mu, logvar = self.encode_distribution(images)
        latent = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        # True KL(q || N(0, I)) per image (summed over latent dims), averaged over the batch
        # and divided by the pixel count so it is in "nats per pixel", the same per-pixel
        # scale as the mean-squared-error term. kl_beta is then 2 * sigma^2 of an equivalent
        # Gaussian decoder (0.01 -> sigma ~ 0.07 on [0, 1] intensities).
        kl_per_image = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).sum(dim=1)
        kl_loss = kl_per_image.mean() / self.num_pixels
        return {"reconstruction": self.decode(latent), "latent": latent, "kl_loss": kl_loss}


class VectorQuantizer(nn.Module):
    """Nearest-neighbour vector quantizer with a straight-through estimator.

    Codebook collapse (a few codes doing all the work) is the usual VQ-VAE failure, so by
    default the codebook is trained the way the original paper recommends and Jukebox-style
    VQ models do:
      * exponential-moving-average (EMA) updates instead of a gradient codebook loss;
      * data-dependent initialisation from the first batch of encoder outputs (the tiny
        +-1/K uniform start is far from where the encoder output actually lives);
      * dead-code restarts: a code whose EMA usage drops below `dead_code_threshold` is
        re-seeded with a random encoder output from the current batch.
    ema=False restores the plain gradient-trained codebook.
    """

    def __init__(
        self,
        num_embeddings: int = 256,
        embedding_dim: int = 64,
        commitment_beta: float = 0.25,
        ema: bool = True,
        ema_decay: float = 0.99,
        dead_code_threshold: float = 0.5,
    ):
        super().__init__()
        if num_embeddings < 2:
            raise ValueError("num_embeddings must be at least 2")
        if commitment_beta < 0:
            raise ValueError("commitment_beta cannot be negative")
        if not 0.0 < ema_decay < 1.0:
            raise ValueError("ema_decay must be in (0, 1)")
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.commitment_beta = float(commitment_beta)
        self.ema = bool(ema)
        self.ema_decay = float(ema_decay)
        self.dead_code_threshold = float(dead_code_threshold)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
        if self.ema:
            self.embedding.weight.requires_grad_(False)  # updated by EMA, not by the optimiser
            self.register_buffer("cluster_size", torch.ones(self.num_embeddings))
            self.register_buffer("embed_avg", self.embedding.weight.detach().clone())
            self.register_buffer("initialized", torch.zeros((), dtype=torch.long))
        self._ready = False  # python-side cache of `initialized` (avoids a device sync per step)

    @torch.no_grad()
    def _seed_codes(self, flat: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return flat[positions] + 1e-3 * torch.randn(len(positions), self.embedding_dim, device=flat.device)

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, indices: torch.Tensor) -> None:
        flat = flat.detach()
        if not self._ready:  # data-dependent start from random encoder outputs
            self._ready = bool(self.initialized)
        if not self._ready:
            positions = torch.randint(0, flat.shape[0], (self.num_embeddings,), device=flat.device)
            seeds = self._seed_codes(flat, positions)
            self.embedding.weight.copy_(seeds)
            self.embed_avg.copy_(seeds)
            self.cluster_size.fill_(1.0)
            self.initialized.fill_(1)
            self._ready = True
            return
        onehot = F.one_hot(indices, self.num_embeddings).to(flat.dtype)  # [N, K]
        decay = self.ema_decay
        self.cluster_size.mul_(decay).add_(onehot.sum(0), alpha=1.0 - decay)
        self.embed_avg.mul_(decay).add_(onehot.t() @ flat, alpha=1.0 - decay)
        # Laplace smoothing keeps rarely used codes from dividing by ~0.
        total = self.cluster_size.sum()
        smoothed = (self.cluster_size + 1e-5) / (total + self.num_embeddings * 1e-5) * total
        self.embedding.weight.copy_(self.embed_avg / smoothed.unsqueeze(1))
        dead = self.cluster_size < self.dead_code_threshold
        if bool(dead.any()):
            positions = torch.randint(0, flat.shape[0], (int(dead.sum()),), device=flat.device)
            seeds = self._seed_codes(flat, positions)
            self.embedding.weight[dead] = seeds
            self.embed_avg[dead] = seeds
            self.cluster_size[dead] = 1.0

    def forward(self, z_e: torch.Tensor):
        """Return straight-through latents, codebook loss, commitment loss, code IDs, perplexity, active codes."""
        if z_e.ndim != 4 or z_e.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected [batch, {self.embedding_dim}, height, width], got {tuple(z_e.shape)}"
            )
        # BCHW -> (B*H*W, D): every spatial position is quantised independently.
        vectors = z_e.permute(0, 2, 3, 1).contiguous()
        flat = vectors.view(-1, self.embedding_dim)
        if self.ema and self.training and not self._ready and not bool(self.initialized):
            self._ema_update(flat, None)  # initialises the codebook from this batch
        codebook = self.embedding.weight
        # Squared L2 distance to every code via |a|^2 + |b|^2 - 2ab (avoids a huge pairwise tensor).
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            + codebook.pow(2).sum(dim=1).unsqueeze(0)
            - 2.0 * flat @ codebook.t()
        )
        indices = distances.argmin(dim=1)  # nearest code per position
        if self.ema and self.training:
            self._ema_update(flat, indices)
            codebook = self.embedding.weight
        quantized = self.embedding(indices).view_as(vectors).permute(0, 3, 1, 2).contiguous()

        # Under EMA the codebook has no gradient, so this is just a quantisation-error monitor.
        codebook_loss = F.mse_loss(quantized, z_e.detach())
        commitment_loss = self.commitment_beta * F.mse_loss(z_e, quantized.detach())
        # Straight-through estimator: the forward pass uses the code, the backward pass
        # copies decoder gradients to the encoder. Codebook gradients come from codebook_loss.
        quantized_st = z_e + (quantized - z_e).detach()

        # Perplexity = exp(entropy of code usage) ~ number of codes in effective use;
        # a value far below codebook_size signals codebook collapse.
        counts = torch.zeros(self.num_embeddings, device=indices.device, dtype=z_e.dtype)
        counts.scatter_add_(0, indices, torch.ones_like(indices, dtype=z_e.dtype))
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
        stages: int = 3,
        ema: bool = True,
        ema_decay: float = 0.99,
        dead_code_threshold: float = 0.5,
    ):
        super().__init__()
        self.encoder = ConvEncoder(base_channels=base_channels, latent_dim=latent_dim, stages=stages)
        self.quantizer = VectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=latent_dim,
            commitment_beta=commitment_beta,
            ema=ema,
            ema_decay=ema_decay,
            dead_code_threshold=dead_code_threshold,
        )
        self.decoder = ConvDecoder(base_channels=base_channels, latent_dim=latent_dim, stages=stages)
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
        # Keep only weights on earlier positions in raster order: rows above the centre
        # and pixels left of the centre. Type A also hides the centre pixel itself (used
        # on the first layer so a position cannot see its own code); type B may see it.
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
            x = x + F.relu(layer(x))  # residual connections ease optimisation of the stack
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
        """Sample [N,H,W] code grids one position at a time (raster order).

        Each step re-runs the network on the partially filled grid; positions not yet
        sampled are zeros, which the masks guarantee are never read.
        """
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
    """Instantiate a model from a params/checkpoint dictionary (see params.TrainParams).

    Keys added later (downsample_stages, codebook_ema, ...) fall back to the original
    behaviour when absent, so checkpoints written before they existed still load.
    """
    base_channels = int(config["base_channels"])
    stages = int(config.get("downsample_stages", 3))
    if model_type == "vae":
        return VAE(
            base_channels=base_channels,
            latent_dim=int(config["vae_latent_dim"]),
            kl_beta=float(config["kl_beta"]),
            image_size=(int(config["image_height"]), int(config["image_width"])),
            stages=stages,
        )
    if model_type == "vqvae":
        return VQVAE(
            base_channels=base_channels,
            latent_dim=int(config["latent_dim"]),
            codebook_size=int(config["codebook_size"]),
            commitment_beta=float(config["commitment_beta"]),
            stages=stages,
            ema=bool(config.get("codebook_ema", 0)),
            ema_decay=float(config.get("ema_decay", 0.99)),
            dead_code_threshold=float(config.get("dead_code_threshold", 0.5)),
        )
    raise ValueError(f"Unknown model_type={model_type!r}; use 'vae' or 'vqvae'.")


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
