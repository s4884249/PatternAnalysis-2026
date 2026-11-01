"""Figure of the final VAE and VQ-VAE architectures with per-block parameter counts.

    python3 architecture_figure.py                       # images/final_architecture.png + printed breakdown
    python3 architecture_figure.py --codebook-size 256   # any params.py override changes the figure

Settings are the ones produced by the documented commands:
    python3 train.py --model vae   --data-root ...                      (params.py defaults)
    python3 train.py --model vqvae --data-root ... --ssim-weight 0.1    (defaults + ssim_weight 0.1)
Parameter counts follow modules.py layer by layer; `train.py` prints the same totals ("Model parameters: N")
and `run_summary.json` stores them (model_parameters, generation.prior_parameters). Only stdlib + matplotlib needed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

from params import TrainParams, add_overrides, apply_overrides


# ----------------------------------------------------------------------------- parameter counting
# Parameter count of a Conv2d / ConvTranspose2d with a k x k kernel: i*o*k*k weights plus o biases.
def conv(i, o, k): return i * o * k * k + o            # Conv2d / ConvTranspose2d with bias
# Parameter count of an affine GroupNorm over c channels (one scale and one shift per channel).
def gn(c): return 2 * c                                 # GroupNorm (affine)
# Parameter count of a Linear layer: i*o weights, plus o biases when `bias` is True.
def lin(i, o, bias=True): return i * o + (o if bias else 0)
# Parameter count of one ResBlock from modules.py: two (GroupNorm, SiLU, 3x3 conv) units. SiLU has no
# parameters.
def res(c): return 2 * conv(c, c, 3) + 2 * gn(c)        # ResBlock: 2 x (GN, SiLU, 3x3 conv)


def vae_counts(p: TrainParams) -> dict:
    """Per-block trainable-parameter counts of the baseline VAE, following modules.VAE layer by layer.

    Returns a dict with per-stage lists ('enc_blocks', 'dec_blocks'), single-layer counts, the flattened
    feature size ('flat') and the grand 'total'. The result feeds both the drawing and the printed
    summary.
    """
    s, b = p.downsample_stages, p.base_channels
    fc = 32  # VAE.feature_channels
    # Spatial size after `s` stride-2 convolutions is H/2^s x W/2^s; the feature map is flattened into
    # the posterior Linear.
    flat = fc * (p.image_height // 2**s) * (p.image_width // 2**s)
    widths = [b * 2**i for i in range(s)]
    # Each encoder block is Conv 4x4 stride 2 + GroupNorm. The first takes the 1-channel image, later
    # ones take the previous block's width.
    enc = [conv(1 if i == 0 else widths[i - 1], w, 4) + gn(w) for i, w in enumerate(widths)]
    c = {"enc_blocks": enc, "enc_conv": conv(widths[-1], fc, 3), "to_posterior": lin(flat, 2 * p.vae_latent_dim),
         "from_latent": lin(p.vae_latent_dim, flat), "dec_conv": conv(fc, widths[-1], 3)}
    # The decoder mirrors the encoder: ConvTranspose blocks run from the widest width down to widths[0];
    # the final ConvTranspose (dec_out) outputs 1 channel.
    c["dec_blocks"] = [conv(widths[i], widths[i - 1], 4) + gn(widths[i - 1]) for i in range(s - 1, 0, -1)]
    c["dec_out"] = conv(widths[0], 1, 4)
    c["total"] = sum(enc) + c["enc_conv"] + c["to_posterior"] + c["from_latent"] + c["dec_conv"] + sum(c["dec_blocks"]) + c["dec_out"]
    c["flat"] = flat
    return c


def vq_counts(p: TrainParams) -> dict:
    """Per-block trainable-parameter counts of the VQ-VAE autoencoder (encoder + decoder), following
    modules.py.

    The EMA codebook is not trainable, so it is reported separately ('codebook_untrained') and left out
    of 'total'; 'ema_buffers' lists the extra non-trainable state the EMA update keeps.
    """
    s, b, D, K, R = p.vq_downsample_stages, p.base_channels, p.latent_dim, p.codebook_size, p.res_blocks
    # VQ widths start at 2 x base_channels (see modules._vq_widths); the plain VAE starts at
    # base_channels.
    widths = [b * 2 ** (i + 1) for i in range(s)]
    c = {"enc_in": conv(1, b, 3) + res(b)}
    c["enc_stages"] = [conv(b if i == 0 else widths[i - 1], w, 4) + res(w) for i, w in enumerate(widths)]
    c["enc_out"] = R * res(widths[-1]) + gn(widths[-1]) + conv(widths[-1], D, 1)
    c["dec_in"] = conv(D, widths[-1], 3) + R * res(widths[-1])
    # Decoder output widths per upsampling stage: the encoder widths in reverse, finishing at
    # base_channels (e.g. [64, 32] for widths [64, 128]).
    outs = list(reversed(widths[:-1])) + [b]
    prev, c["dec_stages"] = widths[-1], []
    for o in outs:
        c["dec_stages"].append(conv(prev, o, 3) + res(o)); prev = o
    c["dec_out"] = gn(b) + conv(b, 1, 3)
    c["total"] = c["enc_in"] + sum(c["enc_stages"]) + c["enc_out"] + c["dec_in"] + sum(c["dec_stages"]) + c["dec_out"]
    c["codebook_untrained"] = K * D          # EMA codebook: requires_grad=False, so NOT in the trainable count
    c["ema_buffers"] = K * D + K + 1         # embed_avg, cluster_size, initialized
    return c


def prior_counts(p: TrainParams) -> dict:
    """Parameter counts of the transformer code prior (modules.CodePrior): embeddings, one block, all
    blocks, output head, plus the code-grid size ('grid') and the 'total'.
    """
    d, K, L = p.prior_hidden, p.codebook_size, p.prior_layers
    # Code-grid size = image size divided by the VQ downsampling factor (2^stages).
    grid = (p.image_height // 2**p.vq_downsample_stages, p.image_width // 2**p.vq_downsample_stages)
    block = 2 * 2 * d + lin(d, 3 * d) + lin(d, d) + lin(d, 4 * d) + lin(4 * d, d)  # 2 LayerNorm + qkv + proj + 2 MLP linears
    # embeddings = token table (K x d) + BOS vector (d) + learned row (H' x d) and column (W' x d)
    # position tables.
    # head = final LayerNorm (2d) + bias-free Linear d -> K.
    c = {"embeddings": K * d + d + grid[0] * d + grid[1] * d, "block": block, "blocks": L * block, "head": 2 * d + d * K, "grid": grid}
    c["total"] = c["embeddings"] + c["blocks"] + c["head"]
    return c


# ----------------------------------------------------------------------------- drawing
GRAY, BLUE, TEAL, ORANGE, RED, GOLD, PURPLE = "#e5e5e5", "#cfe2f3", "#cdebdf", "#fde0c5", "#f6c9c9", "#fff2b3", "#e0d4f0"


# Format a parameter count compactly: '1.23 M', '4.5 k', or the plain integer for small numbers.
def fmt(n): return f"{n / 1e6:.2f} M" if n >= 1e6 else (f"{n / 1e3:.1f} k" if n >= 1e3 else str(n))


def chain(ax, y, items, x0=2.0, x1=98.0, h=9.0):
    """Draw boxes left to right; returns [(x, w)] per box. item = (title, subtitle, colour, params or None)."""
    n = len(items)
    gap = 3.2
    # Box width chosen so that n boxes plus (n - 1) gaps exactly fill the span x0..x1.
    w = (x1 - x0 - gap * (n - 1)) / n
    spots = []
    for k, (title, sub, color, params) in enumerate(items):
        x = x0 + k * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.8", fc=color, ec="#333", lw=1))
        ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center", fontsize=8, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=7.2)
        # The parameter count (when given) is printed just below the box.
        if params is not None:
            ax.text(x + w / 2, y - 1.5, params, ha="center", va="center", fontsize=7.5, color="#555")
        if k:
            # Arrow from the previous box's right edge to this box's left edge (empty text: only
            # the arrow is drawn).
            ax.annotate("", xy=(x - 0.1, y + h / 2), xytext=(x - gap + 0.1, y + h / 2), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#333"))
        spots.append((x, w))
    return spots


def elbow(ax, x_from, y_from, x_to, y_to, y_mid, label=None):
    """Draw an elbow connector: a line from (x_from, y_from) to height y_mid, across to x_to, then an arrow
    pointing into (x_to, y_to). An optional italic label is written just above the horizontal run.
    """
    ax.plot([x_from, x_from, x_to], [y_from, y_mid, y_mid], color="#333", lw=1.2)
    ax.annotate("", xy=(x_to, y_to), xytext=(x_to, y_mid), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#333"))
    if label:
        ax.text((x_from + x_to) / 2, y_mid + 0.9, label, ha="center", fontsize=7, style="italic", color="#444")


def draw(p_vae: TrainParams, p_vq: TrainParams, out: Path) -> None:
    """Render the three-row architecture figure and save it to `out`.

    Row 1: VAE encoder / decoder. Row 2: VQ-VAE encoder / quantiser / decoder. Row 3: the transformer
    prior that generates code maps for the VQ-VAE decoder. `p_vae` and `p_vq` are the settings of the
    two final models (they differ only in ssim_weight by default).
    """
    H, W = p_vq.image_height, p_vq.image_width
    v, q, pr = vae_counts(p_vae), vq_counts(p_vq), prior_counts(p_vq)
    fig, ax = plt.subplots(figsize=(21, 15.5))
    # Work in an abstract 0-100 x 0-112 coordinate system, so every layout number below is relative
    # rather than in pixels.
    ax.set_xlim(0, 100); ax.set_ylim(0, 112); ax.axis("off")
    h = 9.0

    # ---------------- VAE
    s, b, fc, L = p_vae.downsample_stages, p_vae.base_channels, 32, p_vae.vae_latent_dim
    ax.text(2, 109.5, f"VAE (baseline): one global Gaussian latent vector   |   trainable parameters {v['total']:,}  ({fmt(v['total'])})", fontsize=12, fontweight="bold")
    ax.text(2, 106.6, f"loss = MSE + {p_vae.kl_beta:g} · KL/pixels   (ssim_weight {p_vae.ssim_weight:g})   |   latent z ∈ R^{L}; eval uses z = μ; generation: full-covariance Gaussian fitted to training latents", fontsize=8.5, color="#333")
    widths = [b * 2**i for i in range(s)]
    enc = [("Input slice", f"1×{H}×{W}", GRAY, None)]
    for i, w_ in enumerate(widths):
        enc.append(("Conv 4×4 s2\nGN · SiLU", f"{w_}×{H // 2**(i + 1)}×{W // 2**(i + 1)}", BLUE, fmt(v["enc_blocks"][i])))
    hs, ws = H // 2**s, W // 2**s
    enc += [("Conv 3×3", f"{fc}×{hs}×{ws}\nflatten → {v['flat']}", BLUE, fmt(v["enc_conv"])),
            ("Linear\n→ μ, log σ²", f"2 × {L}", ORANGE, fmt(v["to_posterior"]))]
    dec = [("z = μ + σ·ε", f"{L}-d vector", GOLD, None), ("Linear\nreshape", f"{fc}×{hs}×{ws}", ORANGE, fmt(v["from_latent"])),
           ("Conv 3×3\nSiLU", f"{widths[-1]}×{hs}×{ws}", BLUE, fmt(v["dec_conv"]))]
    for k, i in enumerate(range(s - 1, 0, -1)):
        dec.append(("ConvT 4×4 s2\nGN · SiLU", f"{widths[i - 1]}×{H // 2**(i)}×{W // 2**(i)}", BLUE, fmt(v["dec_blocks"][k])))
    dec.append(("ConvT 4×4 s2\nsigmoid  = output", f"1×{H}×{W}", BLUE, fmt(v["dec_out"])))
    ye, yd = 93.0, 77.0
    # chain() returns each box's (x, width), which the connector arrows below use as anchor points.
    a = chain(ax, ye, enc); d = chain(ax, yd, dec)
    # Connector from the end of the encoder row (right) back to the start of the decoder row (left).
    elbow(ax, a[-1][0] + a[-1][1] - 2, ye - 3.0, d[0][0] + 2, yd + h, 87.0, "latent")

    # ---------------- VQ-VAE
    s, b, D, K, R = p_vq.vq_downsample_stages, p_vq.base_channels, p_vq.latent_dim, p_vq.codebook_size, p_vq.res_blocks
    hs, ws = H // 2**s, W // 2**s
    total_q = q["total"] + pr["total"]
    ax.text(2, 70.5, f"VQ-VAE (chosen model): {hs}×{ws} grid of discrete codes + transformer prior   |   autoencoder {q['total']:,} + prior {pr['total']:,} trainable parameters  ({fmt(total_q)} total)", fontsize=12, fontweight="bold")
    ax.text(2, 67.6, f"loss = MSE + {p_vq.ssim_weight:g}·(1 − SSIM) + {p_vq.commitment_beta:g}·commitment (+ quantisation-error monitor, no gradient)   |   EMA codebook (decay {p_vq.ema_decay:g}, dead-code restarts)   |   codebook K={K}, D={D}: {q['codebook_untrained']:,} values, not in the trainable count", fontsize=8.5, color="#333")
    widths = [b * 2 ** (i + 1) for i in range(s)]
    enc = [("Input slice", f"1×{H}×{W}", GRAY, None), ("Conv 3×3\n+ ResBlock", f"{b}×{H}×{W}", TEAL, fmt(q["enc_in"]))]
    for i, w_ in enumerate(widths):
        enc.append(("Conv 4×4 s2\n+ ResBlock", f"{w_}×{H // 2**(i + 1)}×{W // 2**(i + 1)}", TEAL, fmt(q["enc_stages"][i])))
    enc += [(f"{R}× ResBlock\nGN·SiLU·Conv 1×1", f"z_e: {D}×{hs}×{ws}", TEAL, fmt(q["enc_out"])),
            ("Vector quantiser", f"nearest of {K} codes\ncode map {hs}×{ws}", RED, "0 trainable (EMA)")]
    dec = [("z_q (straight-\nthrough)", f"{D}×{hs}×{ws}", GOLD, None), (f"Conv 3×3\n+ {R}× ResBlock", f"{widths[-1]}×{hs}×{ws}", TEAL, fmt(q["dec_in"]))]
    outs = list(reversed(widths[:-1])) + [b]
    for i, o in enumerate(outs):
        dec.append(("Upsample ×2\nConv 3×3 + ResBlock", f"{o}×{hs * 2**(i + 1)}×{ws * 2**(i + 1)}", TEAL, fmt(q["dec_stages"][i])))
    dec.append(("GN·SiLU·Conv 3×3\nsigmoid = output", f"1×{H}×{W}", TEAL, fmt(q["dec_out"])))
    ye, yd = 55.0, 39.0
    a = chain(ax, ye, enc); d = chain(ax, yd, dec)
    elbow(ax, a[-1][0] + a[-1][1] - 2, ye - 3.0, d[0][0] + 2, yd + h, 49.0, "quantised latent")

    # ---------------- prior
    T = pr["grid"][0] * pr["grid"][1]
    # Same rule as modules.CodePrior: attention heads have size 32, so heads = width / 32.
    heads = max(1, p_vq.prior_hidden // 32)
    ax.text(2, 27.0, f"Generation: causal transformer prior over the {pr['grid'][0]}×{pr['grid'][1]} code grid ({T} tokens)   |   trained after the autoencoder with AdamW (lr {p_vq.prior_lr:g}, up to {p_vq.prior_epochs} epochs, early stop on validation cross-entropy)", fontsize=10, fontweight="bold")
    items = [("Code map", f"{pr['grid'][0]}×{pr['grid'][1]} → {T} tokens\nraster order", GRAY, None),
             ("Embeddings", f"token + row + col\nBOS shifted in, d={p_vq.prior_hidden}", ORANGE, fmt(pr["embeddings"])),
             (f"{p_vq.prior_layers}× Transformer block", f"causal attention ({heads} heads)\nMLP {4 * p_vq.prior_hidden}, dropout {p_vq.prior_dropout:g}", PURPLE, f"{fmt(pr['blocks'])} ({fmt(pr['block'])} each)"),
             ("LayerNorm\nLinear d → K", f"logits over {K} codes", ORANGE, fmt(pr["head"])),
             ("Sample next code", "temperature, top-k,\nkey/value cache", GOLD, None)]
    yp = 12.0
    pch = chain(ax, yp, items)
    # The sampled code map is fed back into the decoder row, closing the generation loop.
    elbow(ax, pch[-1][0] + pch[-1][1] - 2, yp + h, d[0][0] + 2, yd, 33.0, "sampled code map → embedding lookup → decoder")

    # Legend built from (colour, meaning) pairs; Patch objects are used because the boxes are drawn as
    # patches, not plotted lines.
    ax.legend(handles=[Patch(fc=c, ec="#333", label=l) for c, l in [(GRAY, "input / data"), (BLUE, "plain conv block"), (TEAL, "residual conv block"), (ORANGE, "linear / embedding"),
                                                                     (RED, "vector quantiser"), (PURPLE, "transformer"), (GOLD, "latent / sampling")]],
              loc="lower center", ncol=7, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.005))
    ax.text(50, 4.3, "Grey numbers under each block are its trainable parameters; tensor shapes are channels × height × width for one image.", ha="center", fontsize=8.5, color="#555")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """CLI entry point: parse the params.py overrides, draw the figure, then print the parameter breakdown
    (a JSON summary plus the per-part dictionaries).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("images/final_architecture.png"))
    add_overrides(parser, TrainParams())  # same overrides as train.py; applied to both models
    args = parser.parse_args()
    base = apply_overrides(TrainParams(), args)
    p_vae = base  # python3 train.py --model vae
    p_vq = base if args.ssim_weight is not None else replace(base, ssim_weight=0.1)  # python3 train.py --model vqvae --ssim-weight 0.1
    draw(p_vae, p_vq, args.out)
    v, q, pr = vae_counts(p_vae), vq_counts(p_vq), prior_counts(p_vq)
    summary = {"vae_trainable": v["total"], "vqvae_autoencoder_trainable": q["total"], "vqvae_codebook_untrained": q["codebook_untrained"],
               "vqvae_prior_trainable": pr["total"], "vqvae_total_trainable": q["total"] + pr["total"]}
    print(json.dumps(summary, indent=2))
    print("VAE parts:", {k: val for k, val in v.items() if k != "total"})
    print("VQ-VAE parts:", {k: val for k, val in q.items() if k != "total"})
    print("Prior parts:", pr)
    print(f"Figure: {args.out.resolve()}")


if __name__ == "__main__":
    main()
