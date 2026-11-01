"""PyTorch modules: a convolutional VAE and a VQ-VAE with a transformer code prior.

The VAE uses the plain conv encoder/decoder stack. The VQ-VAE uses a deeper residual
encoder/decoder (VQEncoder/VQDecoder) because its quality is bounded by how much detail each
discrete code can carry, and a causal-transformer prior over the code grid for generation.
Depends only on PyTorch and the standard library.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(channels: int) -> nn.GroupNorm:
    """GroupNorm with 8 groups when `channels` is divisible by 8, otherwise one group (which normalises
    over all channels and positions).
    """
    return nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)


class ConvEncoder(nn.Module):
    """Convolutional encoder; each of `stages` stride-2 blocks halves the spatial size (8x for 3).

    GroupNorm (not BatchNorm) keeps behaviour independent of batch size and
    identical in train/eval mode; 8 groups when the channel count allows it.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32, latent_dim: int = 64, stages: int = 3):
        """Build `stages` blocks of (4x4 stride-2 conv, GroupNorm, SiLU) with channel widths base,
        2*base, 4*base, ..., followed by a 3x3 conv that outputs `latent_dim` channels. Only 2 or 3
        stages are supported.
        """
        super().__init__()
        if stages not in (2, 3):
            raise ValueError("stages must be 2 (4x downsampling) or 3 (8x downsampling)")
        layers, previous = [], in_channels
        for i in range(stages):
            # Channel width doubles at every stage (e.g. 32, 64, 128).
            width = base_channels * 2**i
            layers += [nn.Conv2d(previous, width, kernel_size=4, stride=2, padding=1), _norm(width), nn.SiLU(inplace=True)]
            previous = width
        layers.append(nn.Conv2d(previous, latent_dim, kernel_size=3, stride=1, padding=1))
        self.network = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode [B, in_channels, H, W] images into a feature map [B, latent_dim, H/2^stages,
        W/2^stages].
        """
        return self.network(images)


class ConvDecoder(nn.Module):
    """Decoder paired with ConvEncoder; outputs intensities in [0, 1]."""

    def __init__(self, out_channels: int = 1, base_channels: int = 32, latent_dim: int = 64, stages: int = 3):
        """Build the mirror of ConvEncoder: a 3x3 conv, then (stages - 1) blocks of (4x4 stride-2
        ConvTranspose, GroupNorm, SiLU), and a final ConvTranspose to `out_channels` with a sigmoid.
        """
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
        """Decode a feature map [B, latent_dim, h, w] into images [B, out_channels, h*2^stages,
        w*2^stages] in [0, 1].
        """
        return self.network(latent)


class ResBlock(nn.Module):
    """Pre-activation residual block: (norm, SiLU, 3x3 conv) twice, with an identity skip."""

    def __init__(self, channels: int):
        """Create two (GroupNorm, SiLU, 3x3 conv) units that keep the channel count and spatial size
        unchanged.
        """
        super().__init__()
        self.block = nn.Sequential(
            _norm(channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
            _norm(channels), nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return x plus the residual branch (an identity skip connection around the two conv units)."""
        return x + self.block(x)


def _vq_widths(base_channels: int, stages: int) -> list:
    """Channel width of each VQ-VAE stage: base * 2, base * 4, ... (one entry per stride-2 stage)."""
    return [base_channels * 2 ** (i + 1) for i in range(stages)]  # e.g. 32 -> [64, 128]


class VQEncoder(nn.Module):
    """Residual conv encoder for the VQ-VAE: stride-2 convs, a ResBlock at every resolution and
    `res_blocks` more at the bottleneck. The last 1x1 conv emits the vectors that get quantised."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32, latent_dim: int = 64, stages: int = 2, res_blocks: int = 2):
        """Build: 3x3 conv + ResBlock at full resolution; per stage a 4x4 stride-2 conv + ResBlock;
        `res_blocks` extra ResBlocks; then GroupNorm, SiLU and a 1x1 conv that emits `latent_dim`
        channels (the vectors to be quantised).
        """
        super().__init__()
        if stages not in (2, 3):
            raise ValueError("stages must be 2 (4x downsampling) or 3 (8x downsampling)")
        layers = [nn.Conv2d(in_channels, base_channels, 3, padding=1), ResBlock(base_channels)]
        previous = base_channels
        for width in _vq_widths(base_channels, stages):
            layers += [nn.Conv2d(previous, width, kernel_size=4, stride=2, padding=1), ResBlock(width)]
            previous = width
        # Extra residual blocks at the lowest resolution (the bottleneck).
        layers += [ResBlock(previous) for _ in range(res_blocks)]
        layers += [_norm(previous), nn.SiLU(), nn.Conv2d(previous, latent_dim, kernel_size=1)]
        self.network = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images [B, in_channels, H, W] into continuous latents z_e [B, latent_dim, H/2^stages,
        W/2^stages].
        """
        return self.network(images)


class VQDecoder(nn.Module):
    """Mirror of VQEncoder. Upsamples with nearest-neighbour + 3x3 conv (no transposed-conv
    checkerboard artefacts) and ends in a sigmoid so intensities are in [0, 1]."""

    def __init__(self, out_channels: int = 1, base_channels: int = 32, latent_dim: int = 64, stages: int = 2, res_blocks: int = 2):
        """Build: 3x3 conv + `res_blocks` ResBlocks at the code-grid resolution; one (nearest x2
        upsample, 3x3 conv, ResBlock) per stage; then GroupNorm, SiLU, 3x3 conv and a sigmoid.
        """
        super().__init__()
        if stages not in (2, 3):
            raise ValueError("stages must be 2 (4x upsampling) or 3 (8x upsampling)")
        widths = _vq_widths(base_channels, stages)
        previous = widths[-1]
        layers = [nn.Conv2d(latent_dim, previous, 3, padding=1)] + [ResBlock(previous) for _ in range(res_blocks)]
        # One upsampling stage per encoder stage: widths in reverse order, ending at base_channels.
        # Each stage = nearest x2 upsample, 3x3 conv, ResBlock.
        for out in list(reversed(widths[:-1])) + [base_channels]:
            layers += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(previous, out, 3, padding=1), ResBlock(out)]
            previous = out
        layers += [_norm(previous), nn.SiLU(), nn.Conv2d(previous, out_channels, 3, padding=1), nn.Sigmoid()]
        self.network = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode quantised latents [B, latent_dim, h, w] into images [B, out_channels, h*2^stages,
        w*2^stages] in [0, 1].
        """
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
        """Build the conv encoder/decoder and the two Linear layers that connect the flattened feature
        map to the global latent vector.

        image_size is (height, width) and must be divisible by 2^stages; kl_beta weights the KL term
        in the loss (see train.compute_loss).
        """
        super().__init__()
        if kl_beta < 0:
            raise ValueError("kl_beta cannot be negative")
        height, width = int(image_size[0]), int(image_size[1])
        factor = 2**stages
        if height % factor or width % factor:
            raise ValueError(f"image_size must be divisible by {factor}")
        # Shape of the conv feature map just before it is flattened: [32, H/2^stages, W/2^stages].
        self.feature_shape = (self.feature_channels, height // factor, width // factor)
        flat = self.feature_channels * (height // factor) * (width // factor)
        # Here the conv encoder's 'latent_dim' is the feature-map channel count (32), not the global
        # latent size; the Linear layers below map the flattened map to and from the global vector.
        self.encoder = ConvEncoder(base_channels=base_channels, latent_dim=self.feature_channels, stages=stages)
        self.to_posterior = nn.Linear(flat, 2 * latent_dim)  # mean and log-variance
        self.from_latent = nn.Linear(latent_dim, flat)
        self.decoder = ConvDecoder(base_channels=base_channels, latent_dim=self.feature_channels, stages=stages)
        self.latent_dim = int(latent_dim)
        self.kl_beta = float(kl_beta)
        self.num_pixels = height * width

    def encode_distribution(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, log-variance) of the posterior q(z | x), each [B, latent_dim]; the log-
        variance is clamped for numerical safety.
        """
        # The Linear layer emits 2 * latent_dim numbers per image: the first half is the mean, the
        # second half the log-variance.
        mu, logvar = self.to_posterior(self.encoder(images).flatten(1)).chunk(2, dim=1)
        # Clamp so exp(logvar) cannot overflow while keeping a wide useful range.
        return mu, logvar.clamp(min=-10.0, max=8.0)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Posterior mean [B, latent_dim]: the deterministic latent used for diagnostics."""
        return self.encode_distribution(images)[0]

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors [B, latent_dim] into images [B, 1, H, W] in [0, 1]."""
        # Un-flatten the vector back into the conv feature map before the decoder.
        return self.decoder(self.from_latent(latent).view(-1, *self.feature_shape))

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return {'reconstruction', 'latent', 'kl_loss'} for a batch of images.

        In training mode z is sampled from the posterior; in eval mode the mean is used. kl_loss is
        the KL divergence to N(0, I) in nats per pixel (not yet multiplied by kl_beta).
        """
        mu, logvar = self.encode_distribution(images)
        # Reparameterisation trick: z = mu + sigma * eps with sigma = exp(logvar / 2), which keeps
        # sampling differentiable. Eval mode uses the mean for reproducible metrics.
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
        """Create the codebook (nn.Embedding) and, in EMA mode, the buffers that track code usage;
        validates the arguments.
        """
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
        # Tiny +-1/K start; in EMA mode it is replaced by the data-dependent initialisation on the
        # first batch.
        nn.init.uniform_(self.embedding.weight, -1.0 / self.num_embeddings, 1.0 / self.num_embeddings)
        if self.ema:
            self.embedding.weight.requires_grad_(False)  # updated by EMA, not by the optimiser
            self.register_buffer("cluster_size", torch.ones(self.num_embeddings))
            self.register_buffer("embed_avg", self.embedding.weight.detach().clone())
            self.register_buffer("initialized", torch.zeros((), dtype=torch.long))
        self._ready = False  # python-side cache of `initialized` (avoids a device sync per step)

    @torch.no_grad()
    def _seed_codes(self, flat: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Copy the encoder vectors at `positions` and add small Gaussian jitter (std 1e-3) so re-
        seeded codes are not exact duplicates.
        """
        return flat[positions] + 1e-3 * torch.randn(len(positions), self.embedding_dim, device=flat.device)

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, indices: torch.Tensor) -> None:
        """EMA codebook update from one training batch (no gradients).

        flat: encoder vectors [N, D]; indices: the code chosen for each (None on the very first
        call, which only initialises).
        1. First call ever: seed every code from random encoder outputs (data-dependent
        initialisation).
        2. Otherwise: update running per-code counts (cluster_size) and running per-code vector sums
        (embed_avg); each code becomes their ratio, i.e. the running mean of the vectors assigned to
        it.
        3. Dead codes (running count below dead_code_threshold) are re-seeded from random vectors of
        this batch.
        """
        # The statistics are updated outside autograd; no gradient flows through the EMA.
        flat = flat.detach()
        if not self._ready:  # data-dependent start from random encoder outputs
            # Re-read the buffer so a checkpoint saved with initialized=1 is not re-seeded after
            # loading.
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
        # onehot.sum(0) = how often each code was chosen in this batch; onehot.t() @ flat = per-code
        # sum of the vectors assigned to it.
        onehot = F.one_hot(indices, self.num_embeddings).to(flat.dtype)  # [N, K]
        decay = self.ema_decay
        # Exponential moving average: new = decay * old + (1 - decay) * batch statistic.
        self.cluster_size.mul_(decay).add_(onehot.sum(0), alpha=1.0 - decay)
        self.embed_avg.mul_(decay).add_(onehot.t() @ flat, alpha=1.0 - decay)
        # Laplace smoothing keeps rarely used codes from dividing by ~0.
        total = self.cluster_size.sum()
        smoothed = (self.cluster_size + 1e-5) / (total + self.num_embeddings * 1e-5) * total
        # Each code vector = running sum / running count, i.e. the running mean of the encoder
        # vectors assigned to that code.
        self.embedding.weight.copy_(self.embed_avg / smoothed.unsqueeze(1))
        # Codes that are almost never chosen are re-seeded from random encoder outputs of this
        # batch.
        dead = self.cluster_size < self.dead_code_threshold
        if bool(dead.any()):
            positions = torch.randint(0, flat.shape[0], (int(dead.sum()),), device=flat.device)
            seeds = self._seed_codes(flat, positions)
            # In-place writes on the weight, which has requires_grad=False in EMA mode (and runs
            # under no_grad).
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
        # First training batch: initialise the codebook BEFORE computing distances. indices=None
        # because nothing has been assigned yet.
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
        # Look up the chosen code for each position, restore [B, H, W, D], then permute back to
        # channels-first [B, D, H, W].
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
        # Histogram of code usage in this batch: add 1 to counts[k] for every position that chose
        # code k.
        counts.scatter_add_(0, indices, torch.ones_like(indices, dtype=z_e.dtype))
        probabilities = counts / counts.sum().clamp_min(1.0)
        nonzero = probabilities > 0
        perplexity = torch.exp(-(probabilities[nonzero] * probabilities[nonzero].log()).sum())
        ids = indices.view(z_e.shape[0], z_e.shape[2], z_e.shape[3])
        active_codes = nonzero.sum().to(z_e.dtype)
        return quantized_st, codebook_loss, commitment_loss, ids, perplexity, active_codes


class VQVAE(nn.Module):
    """Residual conv VQ-VAE; samples are produced using a learned transformer code-index prior."""

    model_type = "vqvae"

    def __init__(
        self,
        base_channels: int = 32,
        latent_dim: int = 64,
        codebook_size: int = 256,
        commitment_beta: float = 0.25,
        stages: int = 2,
        ema: bool = True,
        ema_decay: float = 0.99,
        dead_code_threshold: float = 0.5,
        res_blocks: int = 2,
    ):
        """Assemble the residual encoder, the vector quantiser and the residual decoder from the given
        sizes and codebook settings.
        """
        super().__init__()
        self.encoder = VQEncoder(base_channels=base_channels, latent_dim=latent_dim, stages=stages, res_blocks=res_blocks)
        self.quantizer = VectorQuantizer(
            num_embeddings=codebook_size,
            embedding_dim=latent_dim,
            commitment_beta=commitment_beta,
            ema=ema,
            ema_decay=ema_decay,
            dead_code_threshold=dead_code_threshold,
        )
        self.decoder = VQDecoder(base_channels=base_channels, latent_dim=latent_dim, stages=stages, res_blocks=res_blocks)
        self.latent_dim = latent_dim
        self.codebook_size = codebook_size

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Return continuous encoder output before quantization."""
        return self.encoder(images)

    def quantize(self, latent: torch.Tensor):
        """Quantise encoder output; returns the tuple from VectorQuantizer.forward (straight-through
        latents, codebook loss, commitment loss, code IDs, perplexity, active codes).
        """
        return self.quantizer(latent)

    def encode_indices(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images straight to their [B, H', W'] integer code map (the training data of the code
        prior).
        """
        z_e = self.encode(images)
        # The index map is all the autoregressive code prior needs.
        return self.quantizer(z_e)[3]

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode (quantised) latents [B, D, H', W'] into images."""
        return self.decoder(latent)

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode a [B,H,W] integer code-index grid."""
        # [B, H, W] ids -> [B, H, W, D] embeddings -> [B, D, H, W], the layout the decoder expects.
        quantized = self.quantizer.embedding(indices).permute(0, 3, 1, 2).contiguous()
        return self.decode(quantized)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run encoder, quantiser and decoder; returns a dict with the reconstruction, continuous
        latent, quantised latent, code indices, codebook / commitment losses, perplexity and number
        of active codes.
        """
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


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with an optional key/value cache for fast sampling."""

    def __init__(self, dim: int, heads: int, dropout: float):
        """Create the joint query/key/value projection and the output projection; `dim` must be
        divisible by `heads`.
        """
        super().__init__()
        if dim % heads:
            raise ValueError("hidden size must be divisible by the number of heads")
        self.heads, self.dropout = heads, float(dropout)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
        """Self-attention over x [B, T, dim] in which each position only sees earlier positions.

        Without a cache the whole sequence is processed at once (training / scoring). With a cache
        dict, x holds only the newest position and its keys / values are appended to the cached ones
        (fast autoregressive sampling).
        """
        batch, length, dim = x.shape
        # One Linear produces q, k and v together: [B, T, 3, heads, head_dim] is permuted to [3, B,
        # heads, T, head_dim] and unpacked into q, k, v.
        q, k, v = self.qkv(x).view(batch, length, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        if cache is None:  # training / scoring: whole sequence at once
            # is_causal=True masks out future positions; attention dropout is only active in
            # training.
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:  # sampling: one new query attends to every cached key
            if "k" in cache:
                # dim=2 is the time axis: append the new key / value to those cached from
                # earlier steps.
                k, v = torch.cat([cache["k"], k], dim=2), torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
            # No mask needed: the single new query is the latest position, so it may attend to
            # every cached key.
            y = F.scaled_dot_product_attention(q, k, v)
        # Merge the heads back together: [B, heads, T, head_dim] -> [B, T, dim].
        return self.proj(y.transpose(1, 2).reshape(batch, length, dim))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: x + dropout(attention(LayerNorm(x))), then x + MLP(LayerNorm(x)) with a
    dim -> 4*dim -> dim GELU MLP.
    """
    def __init__(self, dim: int, heads: int, dropout: float):
        """Create the two LayerNorms, the causal attention, the MLP and the dropout."""
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = CausalSelfAttention(dim, heads, dropout)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cache: dict | None = None) -> torch.Tensor:
        """Apply attention then the MLP, each with a residual connection; `cache` is passed through to
        the attention for sampling.
        """
        x = x + self.drop(self.attention(self.norm1(x), cache))
        return x + self.mlp(self.norm2(x))


class CodePrior(nn.Module):
    """GPT-style causal transformer over the VQ-VAE code grid (raster order).

    Why not a small PixelCNN: a stack of 3x3 masked convolutions only sees a local, blind-spotted
    neighbourhood, so it cannot keep an anatomy-wide layout (body outline, bladder, femoral heads)
    consistent across the grid and its samples are locally plausible but globally incoherent. Self-attention
    lets every code condition on ALL earlier codes.

    Position `i` predicts code `i` from codes `< i` (a learned BOS token is shifted in at the start). Positions
    use learned row + column embeddings, so the model knows the 2D geometry. forward() returns logits
    [B, K, H, W], the same layout as before, so the training loss code is unchanged.
    """

    def __init__(
        self,
        codebook_size: int = 512,
        hidden_channels: int = 256,
        height: int = 32,
        width: int = 16,
        layers: int = 6,
        dropout: float = 0.1,
    ):
        """Create token / BOS / row / column embeddings, `layers` transformer blocks (heads =
        hidden_channels / 32) and the output head.

        height x width is the code-grid size the prior is trained for.
        """
        super().__init__()
        self.codebook_size, self.hidden_channels = int(codebook_size), int(hidden_channels)
        self.height, self.width, self.layers, self.dropout = int(height), int(width), int(layers), float(dropout)
        dim = self.hidden_channels
        self.token_embedding = nn.Embedding(self.codebook_size, dim)
        # Learned start-of-sequence vector, shifted in at position 0 so that position i predicts
        # code i.
        self.bos = nn.Parameter(torch.zeros(1, 1, dim))
        self.row_embedding = nn.Embedding(self.height, dim)
        self.col_embedding = nn.Embedding(self.width, dim)
        positions = torch.arange(self.height * self.width)
        # Raster order: token t sits at row t // W and column t % W. persistent=False keeps these
        # constants out of the state_dict.
        self.register_buffer("rows", positions // self.width, persistent=False)
        self.register_buffer("cols", positions % self.width, persistent=False)
        self.drop = nn.Dropout(self.dropout)
        # heads = dim / 32 (head size 32), which is why prior_hidden must be a multiple of 32.
        self.blocks = nn.ModuleList([TransformerBlock(dim, max(1, dim // 32), self.dropout) for _ in range(self.layers)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, self.codebook_size, bias=False)
        # Apply the GPT-style initialisation recursively to every submodule.
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-style initialisation: N(0, 0.02) for Linear and Embedding weights, zeros for Linear
        biases.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _positions(self) -> torch.Tensor:
        """Learned 2D position embedding [H*W, D]: the row embedding plus the column embedding of every
        raster position.
        """
        return self.row_embedding(self.rows) + self.col_embedding(self.cols)  # [H*W, D]

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Teacher-forced next-code logits [B, K, H, W] for a code map [B, H, W]; the logits at
        position t only depend on codes before t.
        """
        if indices.ndim != 3 or indices.shape[1:] != (self.height, self.width):
            raise ValueError(f"Expected code map [B,{self.height},{self.width}], got {tuple(indices.shape)}")
        batch = indices.shape[0]
        tokens = indices.reshape(batch, -1).long()
        # Teacher forcing: the input at position t is [BOS, c0, ..., c_{T-2}] (shifted right by
        # one), so the output at t predicts c_t from earlier codes only.
        x = torch.cat([self.bos.expand(batch, 1, -1), self.token_embedding(tokens[:, :-1])], dim=1) + self._positions()
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))  # [B, T, K]
        # [B, T, K] -> [B, K, T] -> [B, K, H, W]: the layout F.cross_entropy expects, matching the
        # [B, H, W] code map.
        return logits.transpose(1, 2).reshape(batch, self.codebook_size, self.height, self.width)

    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        height: int,
        width: int,
        device: torch.device,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Sample [N,H,W] code grids in raster order using a key/value cache (one cheap step per code).

        top_k > 0 restricts each draw to the k most likely codes, which removes the rare
        low-probability codes that show up as speckle/blocks in an otherwise coherent sample.
        """
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if (height, width) != (self.height, self.width):
            raise ValueError(f"This prior was trained for a {self.height}x{self.width} grid, not {height}x{width}")
        self.eval()
        positions = self._positions()
        total = self.height * self.width
        tokens = torch.zeros((num_samples, total), dtype=torch.long, device=device)
        # One key/value cache per transformer block; each grows by one position per step.
        caches = [dict() for _ in self.blocks]
        # Step 0 feeds BOS at position 0; later steps feed only the previously sampled code (see the
        # end of the loop), earlier positions live in the caches.
        x = self.bos.expand(num_samples, 1, -1) + positions[0]
        for step in range(total):
            for block, cache in zip(self.blocks, caches):
                x = block(x, cache)
            # Only the newest position matters. Temperature rescales the logits (below 1 =
            # sharper, above 1 = flatter).
            logits = self.head(self.norm(x[:, -1])) / temperature
            if 0 < top_k < self.codebook_size:
                # Value of the k-th largest logit in each row; everything below it is set to
                # -inf so it can never be sampled.
                kth = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            sampled = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator).squeeze(-1)
            tokens[:, step] = sampled
            if step + 1 < total:
                # The sampled code becomes the next input, placed at the NEXT grid position.
                x = (self.token_embedding(sampled) + positions[step + 1])[:, None]
        return tokens.view(num_samples, self.height, self.width)


def latent_stages(model_type: str, config: Dict) -> int:
    """Number of stride-2 stages of the given model (the VQ-VAE has its own setting)."""
    if model_type == "vqvae":
        return int(config.get("vq_downsample_stages", 2))
    return int(config.get("downsample_stages", 3))


def build_model(model_type: str, config: Dict) -> nn.Module:
    """Instantiate a model from a params/checkpoint dictionary (see params.TrainParams).

    Keys added later fall back to defaults when absent. NOTE: VQ-VAE checkpoints from before the
    residual encoder/decoder and transformer prior (vq_downsample_stages, res_blocks) have different
    weights and must be retrained.
    """
    base_channels = int(config["base_channels"])
    stages = latent_stages(model_type, config)
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
            # Older configs without the key fall back to the plain gradient-trained codebook
            # (0).
            ema=bool(config.get("codebook_ema", 0)),
            ema_decay=float(config.get("ema_decay", 0.99)),
            dead_code_threshold=float(config.get("dead_code_threshold", 0.5)),
            res_blocks=int(config.get("res_blocks", 2)),
        )
    raise ValueError(f"Unknown model_type={model_type!r}; use 'vae' or 'vqvae'.")


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
