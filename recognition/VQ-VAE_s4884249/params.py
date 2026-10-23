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
    "kl_beta": "vae",
    "codebook_size": "vqvae",
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
    # 128x80 keeps the aspect ratio of the 256x128 and 256x144 slices (padded, not
    # stretched). Both sides must be divisible by 8 (the encoder downsamples 8x).
    image_height: int = 128
    image_width: int = 80
    base_channels: int = 32
    latent_dim: int = 64
    kl_beta: float = 0.1  # VAE: weight of the KL term against the MSE reconstruction loss
    codebook_size: int = 256  # VQ-VAE: number of discrete codes
    commitment_beta: float = 0.25  # VQ-VAE: weight pulling encoder outputs to their codes
    prior_epochs: int = 15  # VQ-VAE: epochs for the PixelCNN-style code prior
    prior_hidden: int = 64
    prior_lr: float = 2e-4
    workers: int = 4  # DataLoader worker processes
    limit_batches: int = 0  # Debug only: cap batches per epoch/evaluation (0 = all)
    device: str = "auto"  # auto, cpu, cuda or cuda:0

    def __post_init__(self):
        if min(self.epochs, self.batch_size) < 1 or self.workers < 0:
            raise ValueError("epochs and batch_size must be >= 1 and workers >= 0")
        if self.kl_beta < 0:
            raise ValueError("kl_beta cannot be negative")
        if self.prior_epochs < 1:
            raise ValueError("prior_epochs must be >= 1 (the VQ-VAE needs a code prior)")


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
            parts.append(f"-{field.name.replace('_', '-')}x{value:g}")
    return "".join(parts)


def run_dir_name(model: str, params: TrainParams) -> str:
    """runs/{model}_seed{seed}x{epochs}[-param]x{value}..."""
    return f"{model}_seed{params.seed}x{params.epochs}{changed_suffix(params, model)}"
