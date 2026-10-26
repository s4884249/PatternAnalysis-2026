"""Central settings for training, prediction and experiments.

Edit the defaults here. Every field can also be overridden on the command line
(e.g. ``--epochs 5 --codebook-size 32``). Overrides that differ from the defaults
are appended to the run directory name, so each configuration gets its own folder:

    runs/vqvae_seed42x50-codebook-sizex32-workersx2
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

MODELS = ("vae", "vqvae")
MANIFEST = Path("manifest.csv")
RUNS_DIR = Path("runs")
EXPERIMENTS_DIR = Path("experiments")

# Parameters that only affect one model. They are ignored (and left out of the
# run name) for the other model, so identical runs never get different folders.
MODEL_SPECIFIC = {
    "vae_latent_dim": "vae",
    "kl_beta": "vae",
    "latent_dim": "vqvae",
    "codebook_size": "vqvae",
    "codebook_ema": "vqvae",
    "ema_decay": "vqvae",
    "dead_code_threshold": "vqvae",
    "prior_patience": "vqvae",
    "commitment_beta": "vqvae",
    "prior_epochs": "vqvae",
    "prior_hidden": "vqvae",
    "prior_lr": "vqvae",
}

# Already encoded in the run name prefix (seed, epochs) or irrelevant to results (device).
NAME_EXCLUDE = {"seed", "epochs", "device"}


@dataclass(frozen=True)
class TrainParams:
    seed: int = 42
    epochs: int = 50
    batch_size: int = 64
    lr: float = 2e-4  # Adam learning rate for the VAE / VQ-VAE
    # Only 256x128 slices are used; 128x64 is exactly 2x down with the same 2:1 ratio
    # (no padding, no centring). Height must stay 2x width and both divisible by 8.
    image_height: int = 128
    image_width: int = 64
    base_channels: int = 32
    latent_dim: int = 64  # VQ-VAE: channels (embedding size) of each code in the 16x8 code grid
    vae_latent_dim: int = 256  # VAE: size of the single global latent vector
    # VAE: KL weight in nats-per-pixel units, equal to 2*sigma^2 of the implied Gaussian
    # decoder. 0.01 -> sigma ~ 0.07. Raise for smoother/more sampleable, lower for sharper.
    kl_beta: float = 0.01
    # 3 = 8x downsampling (128x64 -> 16x8 latent grid). 2 = 4x (32x16 grid): the VQ-VAE keeps
    # 4x more spatial detail (4x more codes per image) at the cost of a slower code prior.
    downsample_stages: int = 3
    codebook_size: int = 256  # VQ-VAE: number of discrete codes
    commitment_beta: float = 0.25  # VQ-VAE: weight pulling encoder outputs to their codes
    codebook_ema: int = 1  # VQ-VAE: 1 = EMA codebook + data init + dead-code restarts, 0 = gradient codebook
    ema_decay: float = 0.99  # VQ-VAE: EMA decay of code usage / sums
    dead_code_threshold: float = 0.5  # VQ-VAE: codes with EMA usage below this are re-seeded
    prior_epochs: int = 40  # VQ-VAE: max epochs for the code prior (stops earlier on validation CE)
    prior_patience: int = 5  # VQ-VAE: stop the prior after this many epochs without a better validation CE (0 = off)
    prior_hidden: int = 64
    prior_lr: float = 2e-4
    # Loss / schedule. ssim_weight adds weight * (1 - SSIM) to the reconstruction term (same for
    # both models). It is off by default: the baseline is plain MSE, and SSIM is the evaluation metric.
    ssim_weight: float = 0.0
    cosine_lr: int = 1  # 1 = cosine-decay the learning rate to 1% of lr over `epochs`, 0 = constant
    patience: int = 50  # stop when validation SSIM has not improved for this many epochs (0 = never)
    # 1 = decode every slice once into RAM (fast epochs; --workers then sets the loading threads and
    # the DataLoaders use no worker processes). 0 = re-read the NIfTI files every epoch with --workers.
    cache_data: int = 1
    min_nonzero_frac: float = 0.0  # drop slices whose fraction of non-zero voxels is below this (blank slices are always dropped)
    workers: int = 4  # loading threads (cache_data=1) or DataLoader worker processes (cache_data=0)
    limit_batches: int = 0  # Debug only: cap batches per epoch/evaluation (0 = all)
    device: str = "auto"  # auto, cpu, mps, cuda or cuda:0

    def __post_init__(self):
        if min(self.epochs, self.batch_size) < 1 or self.workers < 0:
            raise ValueError("epochs and batch_size must be >= 1 and workers >= 0")
        if self.kl_beta < 0:
            raise ValueError("kl_beta cannot be negative")
        if self.prior_epochs < 1:
            raise ValueError("prior_epochs must be >= 1 (the VQ-VAE needs a code prior)")
        if self.downsample_stages not in (2, 3):
            raise ValueError("downsample_stages must be 2 or 3")
        factor = 2**self.downsample_stages
        if self.image_height % factor or self.image_width % factor:
            raise ValueError(f"image_height and image_width must be divisible by {factor} for downsample_stages={self.downsample_stages}")
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in (0, 1)")
        if min(self.ssim_weight, self.patience, self.prior_patience, self.dead_code_threshold, self.min_nonzero_frac) < 0:
            raise ValueError("ssim_weight, patience, prior_patience, dead_code_threshold and min_nonzero_frac cannot be negative")


@dataclass(frozen=True)
class PredictParams:
    seed: int = 42
    num_samples: int = 64  # generated images
    temperature: float = 1.0  # VQ-VAE prior sampling temperature
    max_reference_images: int = 5000  # training images used as the nearest-neighbour pool
    audit_test_count: int = 512  # held-out images used in the nearest-neighbour audit
    batch_size: int = 64
    device: str = "auto"

    def __post_init__(self):
        if min(self.num_samples, self.max_reference_images, self.audit_test_count, self.batch_size) < 1:
            raise ValueError("counts and batch_size must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be greater than zero")


def add_overrides(parser, defaults) -> None:
    """Expose every field as an optional --flag; None means 'keep the default'."""
    for field in fields(defaults):
        parser.add_argument(
            "--" + field.name.replace("_", "-"),
            type=type(getattr(defaults, field.name)),
            default=None,
            help=f"default: {getattr(defaults, field.name)}",
        )


def apply_overrides(defaults, args):
    """Return a copy of `defaults` with every flag the user actually passed."""
    given = {f.name: getattr(args, f.name) for f in fields(defaults) if getattr(args, f.name) is not None}
    return replace(defaults, **given)


def changed_suffix(params: TrainParams, model: str | None = None) -> str:
    """'-codebook-sizex32-workersx2': every parameter that differs from its default."""
    defaults = TrainParams()
    parts = []
    for field in fields(params):
        owner = MODEL_SPECIFIC.get(field.name)
        if field.name in NAME_EXCLUDE or (model and owner and owner != model):
            continue
        value = getattr(params, field.name)
        if value != getattr(defaults, field.name):
            text = f"{value:g}" if isinstance(value, (int, float)) else str(value).replace("/", "_")
            parts.append(f"-{field.name.replace('_', '-')}x{text}")
    return "".join(parts)


def run_dir_name(model: str, params: TrainParams) -> str:
    """runs/{model}_seed{seed}x{epochs}[-param]x{value}..."""
    return f"{model}_seedx{params.seed}-epochsx{params.epochs}{changed_suffix(params, model)}"
