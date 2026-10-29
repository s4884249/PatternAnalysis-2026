"""Evaluation metrics for the synthetic-MRI audit (NumPy / SciPy / scikit-image only, no PyTorch).

Everything here works on plain arrays so it can be unit-tested and re-used without a GPU:

* distribution fidelity  : Frechet distance, precision / recall / density / coverage (Naeem et al. 2020)
* structural fidelity    : best-match SSIM of a synthetic cohort against reference anatomical slices
* diversity / mode drops : k-means mode occupancy, pairwise SSIM / MSE, code-usage entropy
* memorisation           : nearest-neighbour distances, close-match thresholds, data-copying test
* artefact detectors     : checkerboard (Nyquist-corner) energy, Laplacian sharpness, body-mask integrity
* latent interpolation   : path smoothness
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import ndimage, stats

EPS = 1e-12


# --------------------------------------------------------------------------------------
# Basic helpers
# --------------------------------------------------------------------------------------
def ssim_np(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM of two [H, W] images in [0, 1]; identical settings to utils.ssim_2d."""
    from skimage.metrics import structural_similarity

    return float(
        structural_similarity(
            np.asarray(a, np.float32), np.asarray(b, np.float32),
            data_range=1.0, gaussian_weights=True, use_sample_covariance=False,
        )
    )


def avg_pool(images: np.ndarray, factor: int) -> np.ndarray:
    n, h, w = images.shape
    h2, w2 = h // factor, w // factor
    return images[:, : h2 * factor, : w2 * factor].reshape(n, h2, factor, w2, factor).mean(axis=(2, 4))


def bootstrap_ci(values: Sequence[float], stat=np.mean, n_boot: int = 1000, seed: int = 0) -> Tuple[float, float]:
    """95% percentile-bootstrap confidence interval of `stat`."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats_ = [stat(values[rng.integers(0, len(values), len(values))]) for _ in range(n_boot)]
    return float(np.percentile(stats_, 2.5)), float(np.percentile(stats_, 97.5))


def balanced_subsets(*arrays: np.ndarray, seed: int = 0, cap: int | None = None):
    """Random equal-size subsets (size = smallest input, optionally capped) so metrics are comparable."""
    m = min(len(a) for a in arrays)
    if cap:
        m = min(m, cap)
    rng = np.random.default_rng(seed)
    return [a[np.sort(rng.choice(len(a), m, replace=False))] for a in arrays]


# --------------------------------------------------------------------------------------
# Distribution fidelity: FD, precision / recall / density / coverage
# --------------------------------------------------------------------------------------
def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Frechet distance between Gaussians fitted to two feature sets (FID when features are Inception's).

    Uses the symmetric form tr((S1^1/2 S2 S1^1/2)^1/2), which is numerically stabler than a generic sqrtm.
    """
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Need at least two samples per set")
    mu1, mu2 = a.mean(0), b.mean(0)
    s1, s2 = np.atleast_2d(np.cov(a, rowvar=False)), np.atleast_2d(np.cov(b, rowvar=False))
    w, v = np.linalg.eigh(s1)
    root = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    m = root @ s2 @ root
    tr_sqrt = np.sqrt(np.clip(np.linalg.eigvalsh((m + m.T) / 2), 0, None)).sum()
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(s1) + np.trace(s2) - 2.0 * tr_sqrt)


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d2 = (a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2.0 * a @ b.T
    return np.sqrt(np.clip(d2, 0.0, None))


def prdc(real: np.ndarray, fake: np.ndarray, k: int = 5) -> Dict[str, float]:
    """Precision, recall, density, coverage of `fake` w.r.t. `real` (Naeem et al., ICML 2020).

    precision : share of fake samples that land inside the real manifold      -> sample realism
    recall    : share of real samples that land inside the fake manifold      -> diversity / mode drops
    density   : precision that also rewards fakes in dense real regions (can exceed 1)
    coverage  : share of real samples with a fake neighbour inside their k-NN ball (robust to outliers)
    """
    if len(real) <= k or len(fake) <= k:
        raise ValueError("Need more than k samples per set")
    real_r = np.partition(pairwise_distances(real, real), k, axis=1)[:, k]  # index 0 is the point itself
    fake_r = np.partition(pairwise_distances(fake, fake), k, axis=1)[:, k]
    d = pairwise_distances(real, fake)  # [n_real, n_fake]
    return {
        "precision": float((d <= real_r[:, None]).any(axis=0).mean()),
        "recall": float((d <= fake_r[None, :]).any(axis=1).mean()),
        "density": float((d < real_r[:, None]).sum(axis=0).mean() / k),
        "coverage": float((d.min(axis=1) < real_r).mean()),
    }


def pca_fit(x: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, np.float64)
    mean = x.mean(0)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    return mean, vt[:n_components]


def pca_apply(x: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (np.asarray(x, np.float64) - mean) @ components.T


def pixel_pca_features(reference: np.ndarray, *others: np.ndarray, pool: int = 4, n_components: int = 64):
    """Model-agnostic fallback feature space: PCA of 4x-downsampled pixels, fitted on real reference images."""
    ref = avg_pool(reference, pool).reshape(len(reference), -1)
    mean, comps = pca_fit(ref, n_components)
    return [pca_apply(avg_pool(x, pool).reshape(len(x), -1), mean, comps) for x in (reference, *others)]


# --------------------------------------------------------------------------------------
# Nearest neighbours / structural similarity
# --------------------------------------------------------------------------------------
def nn_topk(query: np.ndarray, ref: np.ndarray, k: int = 1, chunk: int = 256, exclude_self: bool = False):
    """k nearest reference images per query by per-pixel MSE. Returns (mse [N,k], index [N,k]) sorted ascending.

    exclude_self=True assumes query and ref are the same set and masks the diagonal.
    """
    q = np.asarray(query, np.float32).reshape(len(query), -1)
    r = np.asarray(ref, np.float32).reshape(len(ref), -1)
    k = min(k, len(r) - (1 if exclude_self else 0))
    dim = q.shape[1]
    r_sq = (r**2).sum(1)
    out_d = np.empty((len(q), k), np.float32)
    out_i = np.empty((len(q), k), np.int64)
    for start in range(0, len(q), chunk):
        qb = q[start : start + chunk]
        d = (qb**2).sum(1)[:, None] + r_sq[None, :] - 2.0 * qb @ r.T
        np.maximum(d, 0.0, out=d)
        if exclude_self:
            d[np.arange(len(qb)), np.arange(start, start + len(qb))] = np.inf
        idx = np.argpartition(d, k - 1, axis=1)[:, :k]
        dd = np.take_along_axis(d, idx, 1)
        order = np.argsort(dd, axis=1)
        out_i[start : start + len(qb)] = np.take_along_axis(idx, order, 1)
        out_d[start : start + len(qb)] = np.take_along_axis(dd, order, 1) / dim
    return out_d, out_i


def best_match_ssim(query: np.ndarray, ref: np.ndarray, candidates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For every query, the highest SSIM over its candidate reference images (e.g. its top-k MSE neighbours).

    SSIM against every reference slice would be O(N*M) skimage calls, so candidates are pre-selected by
    pixel MSE; the result is a slight under-estimate of the true best-match SSIM, never an over-estimate.
    Returns (best_ssim [N], best_reference_index [N]).
    """
    best, best_idx = np.empty(len(query)), np.empty(len(query), np.int64)
    for i in range(len(query)):
        scores = [ssim_np(query[i], ref[j]) for j in candidates[i]]
        a = int(np.argmax(scores))
        best[i], best_idx[i] = scores[a], candidates[i][a]
    return best, best_idx


def pairwise_ssim(images: np.ndarray, n_pairs: int = 400, seed: int = 0) -> np.ndarray:
    """SSIM of random distinct image pairs within one set (lower = more diverse)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_pairs):
        i, j = rng.choice(len(images), 2, replace=False)
        out.append(ssim_np(images[i], images[j]))
    return np.asarray(out)


def data_copying_test(d_to_train: np.ndarray, d_to_test: np.ndarray) -> Dict[str, float]:
    """Simplified data-copying test (Meehan et al., 2020).

    For each synthetic image: distance to its nearest neighbour in a TRAINING subset vs in the held-out TEST
    set. Both reference sets must have the same size, so that for a model that does not copy, the synthetic
    image is equally likely to be closer to either. `fraction_closer_to_train` ~ 0.5 means no copying;
    values well above 0.5 mean the generator sits closer to the training patients than to unseen ones.
    """
    d_to_train, d_to_test = np.asarray(d_to_train), np.asarray(d_to_test)
    n = len(d_to_train)
    wins = float((d_to_train < d_to_test).sum() + 0.5 * (d_to_train == d_to_test).sum())
    return {
        "fraction_closer_to_train": wins / n,
        "p_value_two_sided": float(stats.binomtest(int(round(wins)), n, 0.5).pvalue),
        "n": int(n),
    }


# --------------------------------------------------------------------------------------
# Mode occupancy
# --------------------------------------------------------------------------------------
def kmeans(x: np.ndarray, k: int, seed: int = 0, iters: int = 60) -> np.ndarray:
    """Plain k-means++ / Lloyd; returns centres [k, D]."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, np.float64)
    centres = [x[rng.integers(len(x))]]
    d2 = ((x - centres[0]) ** 2).sum(1)
    for _ in range(1, k):
        probs = d2 / d2.sum() if d2.sum() > 0 else None
        c = x[rng.choice(len(x), p=probs)]
        centres.append(c)
        d2 = np.minimum(d2, ((x - c) ** 2).sum(1))
    centres = np.stack(centres)
    for _ in range(iters):
        labels = assign_clusters(x, centres)
        new = np.stack([x[labels == j].mean(0) if (labels == j).any() else centres[j] for j in range(k)])
        if np.allclose(new, centres):
            break
        centres = new
    return centres


def assign_clusters(x: np.ndarray, centres: np.ndarray) -> np.ndarray:
    return pairwise_distances(x, centres).argmin(axis=1)


def occupancy(labels: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(labels, minlength=k) / max(len(labels), 1)


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def entropy_perplexity(counts: np.ndarray) -> float:
    """exp(entropy) of a count vector: the effective number of categories in use."""
    p = np.asarray(counts, np.float64)
    p = p[p > 0] / p.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


# --------------------------------------------------------------------------------------
# Artefact detectors (used to pick failures objectively rather than by eye)
# --------------------------------------------------------------------------------------
def _power_spectra(images: np.ndarray) -> np.ndarray:
    _, h, w = images.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    x = (images - images.mean(axis=(1, 2), keepdims=True)) * window
    return np.abs(np.fft.fft2(x)) ** 2


def radial_power_spectrum(images: np.ndarray, n_bins: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """Mean radially-averaged power spectrum. Frequencies in cycles/pixel (Nyquist = 0.5)."""
    _, h, w = images.shape
    power = _power_spectra(images).mean(0)
    r = np.sqrt(np.fft.fftfreq(w)[None, :] ** 2 + np.fft.fftfreq(h)[:, None] ** 2)
    edges = np.linspace(0.0, 0.5, n_bins + 1)
    which = np.digitize(r.ravel(), edges) - 1
    keep = (which >= 0) & (which < n_bins)
    total = np.bincount(which[keep], weights=power.ravel()[keep], minlength=n_bins)
    count = np.maximum(np.bincount(which[keep], minlength=n_bins), 1)
    return (edges[:-1] + edges[1:]) / 2, total / count


def checkerboard_score(images: np.ndarray, cutoff: float = 0.375) -> np.ndarray:
    """Fraction of spectral power in the Nyquist corner (|fx| and |fy| > cutoff cycles/pixel).

    Transposed-convolution / upsampling checkerboards put energy at period ~2 px in both axes, i.e. in
    this corner. Compare against the REAL distribution: real MRI noise also has some energy there.
    """
    _, h, w = images.shape
    power = _power_spectra(images)
    corner = (np.abs(np.fft.fftfreq(h))[:, None] > cutoff) & (np.abs(np.fft.fftfreq(w))[None, :] > cutoff)
    return power[:, corner].sum(1) / np.maximum(power.sum(axis=(1, 2)), EPS)


def sharpness(images: np.ndarray) -> np.ndarray:
    """Variance of the Laplacian per image (higher = sharper, near 0 = blurry)."""
    return np.asarray([ndimage.laplace(img.astype(np.float64)).var() for img in images])


def body_mask_stats(images: np.ndarray, min_area: int = 12) -> Dict[str, np.ndarray]:
    """Otsu body mask statistics: component count, fragmentation and interior-hole fraction.

    fragmentation = 1 - (largest component / all foreground): 0 for one solid body, grows when tissue breaks
    into islands. hole_fraction = interior background pixels enclosed by tissue. Judge against real images.
    """
    from skimage.filters import threshold_otsu

    struct = np.ones((3, 3), bool)
    comps, frag, holes, fg = [], [], [], []
    for img in images:
        try:
            threshold = threshold_otsu(img)
        except ValueError:
            threshold = float(img.max())
        mask = img > threshold
        labels, n = ndimage.label(mask, structure=struct)
        if n == 0:
            comps.append(0); frag.append(1.0); holes.append(0.0); fg.append(0.0)
            continue
        areas = np.bincount(labels.ravel())[1:]
        filled = ndimage.binary_fill_holes(mask)
        comps.append(int((areas >= min_area).sum()))
        frag.append(float(1.0 - areas.max() / areas.sum()))
        holes.append(float((filled & ~mask).sum() / max(filled.sum(), 1)))
        fg.append(float(mask.mean()))
    return {
        "n_components": np.asarray(comps), "fragmentation": np.asarray(frag),
        "hole_fraction": np.asarray(holes), "foreground_fraction": np.asarray(fg),
    }


# --------------------------------------------------------------------------------------
# Interpolation
# --------------------------------------------------------------------------------------
def path_smoothness(frames: np.ndarray) -> Dict[str, float]:
    """Step sizes (MSE) along an interpolation path. jump_ratio = max step / mean step (1 = perfectly even;
    large = an abrupt jump, typical of a latent space with holes)."""
    steps = np.asarray([np.mean((frames[i + 1] - frames[i]) ** 2) for i in range(len(frames) - 1)])
    return {"mean_step": float(steps.mean()), "max_step": float(steps.max()), "jump_ratio": float(steps.max() / max(steps.mean(), EPS))}
