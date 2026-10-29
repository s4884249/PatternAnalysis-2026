"""Turn collected arrays into metrics, figures and EVALUATION_RESULTS.md (NumPy only, no PyTorch).

`analyse(data, out_dir)` expects the dictionary produced by evaluate.collect() (documented there).
Every acceptance threshold used in the verdict is a constant at the top of this file, so the criteria are
fixed BEFORE looking at the results and can be tightened or relaxed in one place.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import eval_metrics as M
import eval_plots as P

# ---- acceptance criteria (set before looking at results; edit here to change the verdict) -------------
SSIM_TARGET = 0.6          # assignment target: SSIM must exceed this
NEAR_DUP_SSIM = 0.95       # best-match SSIM at/above this = visually a duplicate of a training slice
MAX_FLAG_RATE = 0.10       # close-match rate; 0.05 is expected BY CONSTRUCTION for real unseen data
MAX_NEAR_DUP_RATE = 0.01
COPY_MAX_FRACTION = 0.55   # data-copying test passes if fraction closer to train <= this ...
COPY_MIN_P = 0.01          # ... or if the two-sided binomial p-value is >= this
MIN_REL_PRECISION = 0.70   # generated precision / recall / coverage relative to the real-test floor
MIN_REL_RECALL = 0.70
MIN_REL_COVERAGE = 0.70
MAX_EXTRA_MODE_DROPS = 1   # clusters of real data under-represented (<25% of real share) beyond the real floor
MAX_ARTEFACT_RATE = 0.10   # share of samples beyond the real 95th percentile (0.05 expected for real data)
NN_CANDIDATES = 10         # MSE-nearest candidates re-ranked by SSIM


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _ci(values) -> tuple:
    values = np.asarray(values, float)
    lo, hi = M.bootstrap_ci(values, n_boot=500)
    return float(values.mean()), lo, hi


def training_stats(rows: List[dict]) -> dict:
    """Stability evidence from history.csv."""
    def col(key):
        return P._col(rows, key)

    if not rows:
        return {}
    ssim, loss = col("val_ssim"), col("train_loss")
    out = {
        "epochs_run": len(rows),
        "best_epoch": int(col("epoch")[int(np.nanargmax(ssim))]),
        "best_val_ssim": float(np.nanmax(ssim)),
        "final_val_ssim": float(ssim[-1]),
        "val_ssim_jitter_last10": float(np.std(np.diff(ssim[-11:]))) if len(ssim) > 2 else None,
        "all_losses_finite": bool(np.isfinite(loss).all() and np.isfinite(col("val_loss")).all()),
        "loss_spikes": int((np.diff(loss) > 0.5 * np.abs(loss[:-1])).sum()) if len(loss) > 1 else 0,  # epochs where loss rose >50%
    }
    gn, clipped = col("train_grad_norm"), col("train_grad_clipped")
    if gn is not None:
        out.update(grad_norm_mean=float(np.nanmean(gn)), grad_norm_max_epoch_mean=float(np.nanmax(gn)),
                   clip_rate=float(np.nanmean(clipped)) if clipped is not None else None)
    active, ppl = col("val_active_codes"), col("val_perplexity")
    if active is not None:
        out.update(final_active_codes=float(active[-1]), final_perplexity=float(ppl[-1]), min_active_codes_after_epoch3=float(np.nanmin(active[3:])) if len(active) > 3 else float(active.min()))
    return out


def analyse(data: dict, out_dir: Path, seed: int = 0, tsne_points: int = 400, k_modes: int = 10) -> dict:
    out_dir = Path(out_dir)
    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    mt = data["models"]
    models = [m for m in P.MODELS if m in mt]
    train, test = data["train"], data["test"]
    train_groups = np.asarray(data["train_groups"])
    rng = np.random.default_rng(seed)
    R: Dict[str, object] = {"models": models}

    R["setup"] = {
        "image_size": list(data["hw"]), "n_train_pool": len(train), "n_train_total": data.get("n_train_total", len(train)),
        "n_test": len(test), "seeds": data["seeds"], "samples_per_seed": data["samples_per_seed"],
        "n_generated_per_model": {m: len(mt[m]["gen"]) for m in models}, "feature_space": data["feature_name"],
        "temperature": data.get("temperature"), "top_k": data.get("top_k"),
        "codebook_size": mt.get("vqvae", {}).get("codebook_size"),
    }

    # ------------------------------------------------------------------ 1. training pipeline / stability
    hist = {m: mt[m]["history"] for m in models}
    P.plot_training_curves(hist, figs / "fig01_training_curves.png", grad_clip=data.get("grad_clip", 5.0),
                           codebook_size=mt.get("vqvae", {}).get("codebook_size"))
    P.plot_loss_components(hist, figs / "fig02_loss_components.png")
    if mt.get("vqvae", {}).get("prior_history"):
        P.plot_prior_curves(mt["vqvae"]["prior_history"], mt["vqvae"]["codebook_size"], figs / "fig03_prior_training.png")
    R["training"] = {m: training_stats(hist[m]) for m in models}
    R["training_summary"] = {m: {k: mt[m]["run_summary"].get(k) for k in ("best_epoch", "epochs_run", "stopped_early", "training_wall_seconds", "device")} for m in models}
    if mt.get("vqvae", {}).get("prior_history"):
        ph = mt["vqvae"]["prior_history"]
        val = P._col(ph, "val_cross_entropy")
        R["prior"] = {"epochs_run": len(ph), "best_val_cross_entropy": float(np.nanmin(val)) if val is not None else None,
                      "uniform_cross_entropy": float(np.log(mt["vqvae"]["codebook_size"])),
                      "final_train_cross_entropy": float(ph[-1]["train_cross_entropy"])}

    # ------------------------------------------------------------------ 2. reconstructions
    recon_ssim = {m: np.asarray([M.ssim_np(test[i], mt[m]["recon_test"][i]) for i in range(len(test))]) for m in models}
    recon_mse = {m: ((test - mt[m]["recon_test"]) ** 2).mean(axis=(1, 2)) for m in models}
    idx = np.unique(np.linspace(0, len(test) - 1, 8).astype(int))
    P.plot_reconstructions(test, {m: mt[m]["recon_test"] for m in models}, recon_ssim, idx, figs / "fig04_reconstructions.png")
    R["reconstruction"] = {m: {"ssim": _ci(recon_ssim[m]), "ssim_median": float(np.median(recon_ssim[m])), "ssim_min": float(recon_ssim[m].min()),
                               "fraction_above_target": float((recon_ssim[m] > SSIM_TARGET).mean()), "mse_mean": float(recon_mse[m].mean())} for m in models}

    # ------------------------------------------------------------------ 3. samples (uncurated)
    for m in models:
        P.plot_sample_grids(mt[m]["gen"], np.asarray(mt[m]["gen_seed"]), m, figs / f"fig05_samples_{m}.png")
    rows = {"real_train": train, "real_test": test, **{m: mt[m]["gen"] for m in models}}
    P.plot_real_vs_synth(rows, figs / "fig06_real_vs_synthetic.png", seed=seed)

    # ------------------------------------------------------------------ 4. nearest-training-neighbour audit
    sets = {m: mt[m]["gen"] for m in models}
    sets["real_test"] = test
    if mt.get("vqvae", {}).get("recon_control") is not None:
        sets["control"] = mt["vqvae"]["recon_control"]  # reconstructions of TRAINING images: a deliberate "memorised" positive control
    nn: Dict[str, dict] = {}
    for key, imgs in sets.items():
        d, ix = M.nn_topk(imgs, train, k=NN_CANDIDATES)
        best, best_ix = M.best_match_ssim(imgs, train, ix)
        nn[key] = {"d1": d[:, 0], "best_ssim": best, "best_idx": best_ix}
    threshold = float(np.quantile(nn["real_test"]["d1"], 0.05))
    sub = np.sort(rng.choice(len(train), min(len(test), len(train)), replace=False))
    copy, copy_pairs = {}, {}
    for m in models:
        d_train = M.nn_topk(sets[m], train[sub], 1)[0][:, 0]
        d_test = M.nn_topk(sets[m], test, 1)[0][:, 0]
        copy[m] = M.data_copying_test(d_train, d_test)
        copy_pairs[m] = (d_train, d_test)
    R["memorisation"] = {"close_match_threshold_mse": threshold, "near_duplicate_ssim": NEAR_DUP_SSIM}
    for key in nn:
        groups = train_groups[nn[key]["best_idx"]]
        _, counts = np.unique(groups, return_counts=True)
        R["memorisation"][key] = {
            "d1_mean": float(nn[key]["d1"].mean()), "d1_min": float(nn[key]["d1"].min()),
            "flagged_close_match_rate": float((nn[key]["d1"] <= threshold).mean()),
            "best_match_ssim": _ci(nn[key]["best_ssim"]), "best_match_ssim_max": float(nn[key]["best_ssim"].max()),
            "near_duplicate_rate": float((nn[key]["best_ssim"] >= NEAR_DUP_SSIM).mean()),
            "fraction_above_target": float((nn[key]["best_ssim"] > SSIM_TARGET).mean()),
            "distinct_nearest_cases": int(len(counts)), "top_case_share": float(counts.max() / counts.sum()),
        }
    for m in models:
        R["memorisation"][m]["data_copying_test"] = copy[m]
    P.plot_nn_audit({k: nn[k]["d1"] for k in nn}, threshold, {k: nn[k]["best_ssim"] for k in nn}, NEAR_DUP_SSIM, copy_pairs, figs / "fig07_nearest_neighbour_audit.png")
    for m in models:
        order = np.argsort(nn[m]["d1"])[:6]
        trip = []
        for i in order:
            j = int(nn[m]["best_idx"][i])
            trip.append({"image": sets[m][i], "neighbor": train[j],
                         "title": f"#{i} | pixel MSE {nn[m]['d1'][i]:.5f} | SSIM {nn[m]['best_ssim'][i]:.3f} | train case {train_groups[j]}"})
        P.plot_triplets(trip, figs / f"fig08_closest_to_training_{m}.png", f"{P.LABELS[m]}: the 6 generated images closest to any training slice")

    # ------------------------------------------------------------------ 5. distribution fidelity, coverage, diversity
    F = data["features"]
    train_f, test_f, gen_f = F["train"], F["test"], {m: F[m] for m in models}
    fd = {m: M.frechet_distance(gen_f[m], train_f) for m in models}
    fd["real_test"] = M.frechet_distance(test_f, train_f)
    pix = M.pixel_pca_features(train[F["train_idx"]], test, *[sets[m] for m in models])
    fd_pix = {m: M.frechet_distance(pix[2 + i], pix[0]) for i, m in enumerate(models)}
    fd_pix["real_test"] = M.frechet_distance(pix[1], pix[0])
    bal = M.balanced_subsets(train_f, test_f, *[gen_f[m] for m in models], seed=seed, cap=1000)
    prdc = {m: M.prdc(bal[0], bal[2 + i]) for i, m in enumerate(models)}
    prdc["real_test"] = M.prdc(bal[0], bal[1])
    R["fidelity"] = {"feature_name": data["feature_name"], "fd": fd, "fd_pixel_pca": fd_pix, "prdc": prdc, "prdc_n_per_set": len(bal[0]),
                     "note": "FD from a few hundred samples is biased upwards; compare against the real-test floor, not against zero."}
    match_ssim = {m: _ci(nn[m]["best_ssim"]) for m in models}
    match_ssim["real_test"] = _ci(nn["real_test"]["best_ssim"])
    P.plot_metric_bars({"recon_ssim": {m: R["reconstruction"][m]["ssim"] for m in models}, "match_ssim": match_ssim,
                        "fd": fd, "fd_name": data["feature_name"], "prdc": prdc}, figs / "fig09_metric_comparison.png")

    pca_mean, pca_comp = M.pca_fit(train_f, min(32, train_f.shape[1], len(train_f) - 1))
    z = lambda a: M.pca_apply(a, pca_mean, pca_comp)
    centres = M.kmeans(z(train_f), k_modes, seed=seed)
    occ_real = M.occupancy(M.assign_clusters(z(train_f), centres), k_modes)
    occ = {m: M.occupancy(M.assign_clusters(z(gen_f[m]), centres), k_modes) for m in models}
    occ_test = M.occupancy(M.assign_clusters(z(test_f), centres), k_modes)
    pair = {"real_test": M.pairwise_ssim(test, 400, seed)}
    pair.update({m: M.pairwise_ssim(sets[m], 400, seed) for m in models})
    intra = {k: float(M.nn_topk(sets[k], sets[k], 1, exclude_self=True)[0][:, 0].mean()) for k in models + ["real_test"]}
    code_usage = None
    vq = mt.get("vqvae", {})
    R["diversity"] = {
        "k_modes": k_modes,
        "tv_distance_to_real_occupancy": {**{m: M.total_variation(occ[m], occ_real) for m in models}, "real_test": M.total_variation(occ_test, occ_real)},
        "mode_drops": {**{m: int((occ[m] < 0.25 * occ_real).sum()) for m in models}, "real_test": int((occ_test < 0.25 * occ_real).sum())},
        "pairwise_ssim_mean": {k: float(v.mean()) for k, v in pair.items()},
        "intra_cohort_nn_mse": intra,
    }
    if vq.get("codes") is not None and vq.get("train_codes") is not None:
        K = vq["codebook_size"]
        sampled = np.bincount(np.asarray(vq["codes"]).ravel(), minlength=K)
        trained = np.bincount(np.asarray(vq["train_codes"]).ravel(), minlength=K)
        code_usage = {"train": trained, "sampled": sampled}
        R["diversity"]["code_usage"] = {"codebook_size": K, "codes_used_by_train": int((trained > 0).sum()), "codes_used_in_samples": int((sampled > 0).sum()),
                                        "perplexity_train": M.entropy_perplexity(trained), "perplexity_samples": M.entropy_perplexity(sampled)}
    P.plot_diversity(occ_real, occ, pair, code_usage, figs / "fig10_diversity.png")

    # ------------------------------------------------------------------ 6. clarity + artefact detectors
    art = {k: {"sharp": M.sharpness(sets[k]), "checker": M.checkerboard_score(sets[k]), "mask": M.body_mask_stats(sets[k])} for k in models + ["real_test"]}
    chk95 = float(np.quantile(art["real_test"]["checker"], 0.95))
    frag95 = float(np.quantile(art["real_test"]["mask"]["fragmentation"], 0.95))
    sharp_med = float(np.median(art["real_test"]["sharp"]))
    P.plot_clarity({k: M.radial_power_spectrum(sets[k]) for k in models + ["real_test"]}, {k: art[k]["sharp"] for k in art},
                   {k: art[k]["checker"] for k in art}, figs / "fig11_clarity.png")
    R["clarity"] = {"real_checkerboard_p95": chk95, "real_fragmentation_p95": frag95, "real_sharpness_median": sharp_med}
    for k in art:
        R["clarity"][k] = {"sharpness_median": float(np.median(art[k]["sharp"])), "sharpness_ratio_to_real": float(np.median(art[k]["sharp"]) / max(sharp_med, 1e-12)),
                           "checkerboard_flag_rate": float((art[k]["checker"] > chk95).mean()),
                           "fragmentation_flag_rate": float((art[k]["mask"]["fragmentation"] > frag95).mean()),
                           "mean_components": float(art[k]["mask"]["n_components"].mean())}

    # ------------------------------------------------------------------ 7. failure autopsy (objective, automatic selection)
    R["autopsy"] = {}
    for m in models:
        a, n = art[m], len(sets[m])
        knn = np.sort(M.pairwise_distances(gen_f[m], train_f), axis=1)[:, :5].mean(1)
        frag = a["mask"]["fragmentation"] + 0.05 * np.maximum(a["mask"]["n_components"] - 1, 0)
        crit = [
            ("checkerboard / high-frequency artefact", a["checker"], lambda i: f"Nyquist-corner energy {a['checker'][i]:.3g} (real 95th pct {chk95:.3g})"),
            ("fragmented tissue margin", frag, lambda i: f"body mask fragmentation {a['mask']['fragmentation'][i]:.2f}, {a['mask']['n_components'][i]} components (real 95th pct frag {frag95:.2f})"),
            ("off-manifold anatomy", knn, lambda i: f"mean 5-NN feature distance {knn[i]:.3g} (median of cohort {np.median(knn):.3g})"),
            ("blur / lost detail", -a["sharp"], lambda i: f"Laplacian variance {a['sharp'][i]:.3g} (real median {sharp_med:.3g})"),
            ("no close anatomical match", -nn[m]["best_ssim"], lambda i: f"best SSIM to any training slice {nn[m]['best_ssim'][i]:.2f}"),
        ]
        used, trip, table = set(), [], []
        for name, score, describe in crit:
            i = next(int(c) for c in np.argsort(-score) if int(c) not in used)
            used.add(i)
            j = int(nn[m]["best_idx"][i])
            s = int(np.asarray(mt[m]["gen_seed"])[i])
            pos = int(i - np.flatnonzero(np.asarray(mt[m]["gen_seed"]) == s)[0])
            trip.append({"image": sets[m][i], "neighbor": train[j], "title": f"{name}: {describe(i)} | seed {s} #{pos} | nearest train case {train_groups[j]} (SSIM {nn[m]['best_ssim'][i]:.2f})"})
            table.append({"failure_type": name, "seed": s, "index": pos, "evidence": describe(i), "nearest_case": str(train_groups[j]), "nearest_ssim": float(nn[m]["best_ssim"][i])})
        P.plot_triplets(trip, figs / f"fig12_failure_autopsy_{m}.png", f"{P.LABELS[m]}: worst sample per automatic failure criterion (no manual selection)")
        R["autopsy"][m] = table

    # ------------------------------------------------------------------ 8. embedding + interpolation
    pick = lambda arr: arr[np.sort(rng.choice(len(arr), min(tsne_points, len(arr)), replace=False))]
    parts = [("real_train", pick(train_f)), ("real_test", pick(test_f))] + [(m, pick(gen_f[m])) for m in models]
    emb, method = P.embed_2d(np.concatenate([p for _, p in parts]), seed)
    P.plot_embedding(emb, np.concatenate([[name] * len(p) for name, p in parts]), method, data["feature_name"], figs / "fig13_embedding.png")
    R["embedding_method"] = method
    paths = {m: mt[m].get("interp", []) for m in models}
    P.plot_interpolation(paths, figs / "fig14_latent_interpolation.png")
    R["interpolation"] = {m: {"mean_jump_ratio": float(np.mean([M.path_smoothness(p["frames"])["jump_ratio"] for p in paths[m]])) if paths[m] else None,
                              "max_jump_ratio": float(np.max([M.path_smoothness(p["frames"])["jump_ratio"] for p in paths[m]])) if paths[m] else None} for m in models}

    # ------------------------------------------------------------------ 9. resource profile
    prof = {}
    for m in models:
        rs = mt[m]["run_summary"]
        seconds = rs.get("total_training_plus_prior_wall_seconds") or rs.get("training_wall_seconds")
        prof[m] = {"params_m": (mt[m]["n_params"] + (mt[m].get("prior_params") or 0)) / 1e6, "train_minutes": seconds / 60 if seconds else None,
                   "train_peak_mb": rs.get("peak_accelerator_memory_mb") or rs.get("peak_cuda_memory_mb"),
                   "gen_ms_per_sample": 1000 * mt[m]["gen_seconds_per_sample"], "gen_peak_mb": mt[m].get("gen_peak_mb"),
                   "autoencoder_params_m": mt[m]["n_params"] / 1e6, "prior_params_m": (mt[m].get("prior_params") or 0) / 1e6}
    P.plot_resource_profile(prof, figs / "fig15_resource_profile.png")
    R["resources"] = prof

    R["verdict"] = {m: verdict(R, m) for m in models}
    (out_dir / "metrics.json").write_text(json.dumps(_jsonable(R), indent=2) + "\n", encoding="utf-8")
    write_markdown(R, out_dir / "EVALUATION_RESULTS.md")
    return R


# ======================================================================================================
# verdict
# ======================================================================================================
def verdict(R: dict, m: str) -> dict:
    mem, fid, div, cl = R["memorisation"][m], R["fidelity"], R["diversity"], R["clarity"]
    copy = mem["data_copying_test"]
    floor = fid["prdc"]["real_test"]
    rel = lambda key: fid["prdc"][m][key] / max(floor[key], 1e-9)
    crit = [
        ("Fidelity", f"Test reconstruction SSIM > {SSIM_TARGET}", f"{R['reconstruction'][m]['ssim'][0]:.3f}", R["reconstruction"][m]["ssim"][0] > SSIM_TARGET),
        ("Fidelity", f"Synthetic best-match SSIM to real slices > {SSIM_TARGET}", f"{mem['best_match_ssim'][0]:.3f}", mem["best_match_ssim"][0] > SSIM_TARGET),
        ("Memorisation", f"Close-match rate <= {MAX_FLAG_RATE:.0%} (real-data expectation 5%)", f"{mem['flagged_close_match_rate']:.1%}", mem["flagged_close_match_rate"] <= MAX_FLAG_RATE),
        ("Memorisation", f"Near-duplicates (SSIM >= {NEAR_DUP_SSIM}) <= {MAX_NEAR_DUP_RATE:.0%}", f"{mem['near_duplicate_rate']:.1%}", mem["near_duplicate_rate"] <= MAX_NEAR_DUP_RATE),
        ("Memorisation", f"Data-copying test: fraction closer to train <= {COPY_MAX_FRACTION} or p >= {COPY_MIN_P}",
         f"{copy['fraction_closer_to_train']:.2f} (p={copy['p_value_two_sided']:.3g})", copy["fraction_closer_to_train"] <= COPY_MAX_FRACTION or copy["p_value_two_sided"] >= COPY_MIN_P),
        ("Diversity", f"Recall >= {MIN_REL_RECALL:.0%} of real-test floor", f"{rel('recall'):.0%} (recall {fid['prdc'][m]['recall']:.2f})", rel("recall") >= MIN_REL_RECALL),
        ("Diversity", f"Coverage >= {MIN_REL_COVERAGE:.0%} of real-test floor", f"{rel('coverage'):.0%} (coverage {fid['prdc'][m]['coverage']:.2f})", rel("coverage") >= MIN_REL_COVERAGE),
        ("Diversity", f"Mode drops <= real floor + {MAX_EXTRA_MODE_DROPS}", f"{div['mode_drops'][m]} vs {div['mode_drops']['real_test']}", div["mode_drops"][m] <= div["mode_drops"]["real_test"] + MAX_EXTRA_MODE_DROPS),
        ("Plausibility", f"Precision >= {MIN_REL_PRECISION:.0%} of real-test floor", f"{rel('precision'):.0%} (precision {fid['prdc'][m]['precision']:.2f})", rel("precision") >= MIN_REL_PRECISION),
        ("Plausibility", f"Checkerboard-artefact rate <= {MAX_ARTEFACT_RATE:.0%}", f"{cl[m]['checkerboard_flag_rate']:.1%}", cl[m]["checkerboard_flag_rate"] <= MAX_ARTEFACT_RATE),
        ("Plausibility", f"Fragmented-tissue rate <= {MAX_ARTEFACT_RATE:.0%}", f"{cl[m]['fragmentation_flag_rate']:.1%}", cl[m]["fragmentation_flag_rate"] <= MAX_ARTEFACT_RATE),
    ]
    items = [{"group": g, "criterion": c, "value": v, "pass": bool(p)} for g, c, v, p in crit]
    failed = {g: [i["criterion"] for i in items if i["group"] == g and not i["pass"]] for g in ("Fidelity", "Memorisation", "Diversity", "Plausibility")}
    if failed["Memorisation"]:
        label = "NOT SAFE TO SHARE: memorisation criteria failed"
    elif any(failed.values()):
        label = "NO EVIDENCE OF MEMORISATION, but NOT YET FIT FOR DOWNSTREAM TRAINING"
    else:
        label = "CONDITIONALLY SAFE: passes every automatic criterion (expert review still required)"
    return {"label": label, "criteria": items, "failed": failed}


# ======================================================================================================
# markdown
# ======================================================================================================
def _tbl(header: List[str], rows: List[List[str]]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def _ci_text(t) -> str:
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def write_markdown(R: dict, path: Path) -> None:
    models = R["models"]
    L = {"vae": "VAE", "vqvae": "VQ-VAE", "real_test": "Real test (floor)", "control": "Train reconstructions (control)"}
    s, rec, mem, fid, div, cl, tr, res = (R[k] for k in ("setup", "reconstruction", "memorisation", "fidelity", "diversity", "clarity", "training", "resources"))
    F = lambda name, caption: f"![{caption}](figures/{name})\n\n*{caption}*\n"
    out: List[str] = []
    w = out.append

    w("# Evaluation results: VQ-VAE vs VAE baseline on HipMRI slices\n")
    w("*Generated automatically by `evaluate.py`; every number below is read from `metrics.json`. "
      "Architecture and loss definitions are in `IMPLEMENTATION_AND_ARCHITECTURE.md`.*\n")

    w("## 0. Executive verdict\n")
    for m in models:
        v = R["verdict"][m]
        w(f"**{L[m]}: {v['label']}**\n")
    if "vqvae" in models:
        v = R["verdict"]["vqvae"]
        w(_tbl(["Group", "Criterion (fixed in advance)", "VQ-VAE", "Pass"], [[i["group"], i["criterion"], i["value"], "yes" if i["pass"] else "**NO**"] for i in v["criteria"]]) + "\n")
    if "vae" in models:
        v = R["verdict"]["vae"]
        w("Baseline VAE under the same criteria:\n")
        w(_tbl(["Group", "Criterion", "VAE", "Pass"], [[i["group"], i["criterion"], i["value"], "yes" if i["pass"] else "**NO**"] for i in v["criteria"]]) + "\n")
    w("> The verdict is rule-based. Thresholds are constants at the top of `eval_analysis.py`. A nearest-neighbour audit can *fail to find* "
      "memorisation; it cannot prove its absence, and it says nothing about clinical correctness. See Section 6 for what this does and does not establish.\n")

    w("## 1. Setup\n")
    w(f"- Image size {s['image_size'][0]}x{s['image_size'][1]}; reference (training) pool {s['n_train_pool']} slices (of {s['n_train_total']}); held-out test {s['n_test']} slices.\n"
      f"- Synthetic cohorts: {len(s['seeds'])} random seeds {s['seeds']} x {s['samples_per_seed']} samples per model "
      f"(temperature {s['temperature']}, top-k {s['top_k']}; no tuning for the evaluation).\n"
      f"- Feature space for FD / precision / recall / coverage / t-SNE: **{s['feature_space']}**.\n"
      f"- *Real-test floor*: the same metric computed with held-out real slices in place of synthetic ones. A synthetic cohort cannot be expected to beat this; "
      "matching it is the realistic goal.\n")

    w("## 2. Pipeline setup and training stability\n")
    rows = []
    for m in models:
        t = tr[m]
        rows.append([L[m], t.get("epochs_run"), t.get("best_epoch"), f"{t.get('best_val_ssim', float('nan')):.3f}", f"{t['val_ssim_jitter_last10']:.4f}" if t.get("val_ssim_jitter_last10") is not None else "n/a",
                     "yes" if t.get("all_losses_finite") else "**NO**", t.get("loss_spikes"),
                     f"{t['grad_norm_mean']:.2f}" if "grad_norm_mean" in t else "not logged", f"{t['clip_rate']:.1%}" if t.get("clip_rate") is not None else "n/a"])
    w(_tbl(["Model", "Epochs", "Best epoch", "Best val SSIM", "SSIM jitter (last 10 ep, std of delta)", "All losses finite", "Loss spikes (>50% rise)", "Mean grad norm (pre-clip)", "Steps clipped"], rows) + "\n")
    if "vqvae" in tr and "final_active_codes" in tr["vqvae"]:
        t, K = tr["vqvae"], s["codebook_size"]
        w(f"Codebook health (VQ-VAE, validation): {t['final_active_codes']:.0f} of {K} codes active at the last epoch (minimum after epoch 3: {t['min_active_codes_after_epoch3']:.0f}), "
          f"perplexity {t['final_perplexity']:.0f}. Perplexity far below the codebook size would indicate codebook collapse.\n")
    if "prior" in R:
        p = R["prior"]
        w(f"Code prior: best validation cross-entropy {p['best_val_cross_entropy']:.3f} nats/code vs {p['uniform_cross_entropy']:.3f} for a uniform prior; "
          f"trained {p['epochs_run']} epochs (final train CE {p['final_train_cross_entropy']:.3f}). A growing train/validation gap means the prior is memorising code maps; early stopping restores the best epoch.\n")
    w(F("fig01_training_curves.png", "Training curves, gradient norms and codebook usage for both models (star = checkpoint kept by validation SSIM)"))
    w(F("fig02_loss_components.png", "Loss components: VAE reconstruction vs KL; VQ-VAE reconstruction, quantisation error and commitment"))
    if "prior" in R:
        w(F("fig03_prior_training.png", "Transformer code-prior cross-entropy"))

    w("## 3. Baseline vs VQ-VAE: sample clarity, diversity, fidelity\n")
    w(F("fig06_real_vs_synthetic.png", "Random real (train, test) and synthetic images; rows are not curated"))
    w(F("fig11_clarity.png", "Clarity: radial power spectrum (blur removes high frequencies), Laplacian sharpness and Nyquist-corner (checkerboard) energy"))
    w(_tbl(["Clarity", *[L[k] for k in ["real_test", *models]]],
           [["Median Laplacian sharpness (ratio to real)", *[f"{cl[k]['sharpness_ratio_to_real']:.2f}" for k in ["real_test", *models]]],
            ["Checkerboard flag rate (>real p95; real = 5%)", *[f"{cl[k]['checkerboard_flag_rate']:.1%}" for k in ["real_test", *models]]],
            ["Fragmented-tissue flag rate (>real p95; real = 5%)", *[f"{cl[k]['fragmentation_flag_rate']:.1%}" for k in ["real_test", *models]]]]) + "\n")
    w("Sharpness ratio < 1 means blurrier than real MRI; the VQ-VAE is expected to sit closer to 1 than the VAE because it is not forced through a single Gaussian bottleneck.\n")
    w(_tbl(["Diversity", *[L[k] for k in ["real_test", *models]]],
           [["Mean pairwise SSIM between random pairs (lower = more diverse)", *[f"{div['pairwise_ssim_mean'][k]:.3f}" for k in ["real_test", *models]]],
            ["Intra-cohort nearest-neighbour MSE (higher = fewer near-duplicates)", *[f"{div['intra_cohort_nn_mse'][k]:.5f}" for k in ["real_test", *models]]],
            [f"Mode occupancy TV distance to real ({div['k_modes']} clusters; 0 = identical)", *[f"{div['tv_distance_to_real_occupancy'][k]:.3f}" for k in ["real_test", *models]]],
            ["Under-represented clusters (mode drops)", *[str(div['mode_drops'][k]) for k in ["real_test", *models]]]]) + "\n")
    if "code_usage" in div:
        c = div["code_usage"]
        w(f"VQ-VAE codes: training images use {c['codes_used_by_train']}/{c['codebook_size']} codes (perplexity {c['perplexity_train']:.0f}); "
          f"prior samples use {c['codes_used_in_samples']}/{c['codebook_size']} (perplexity {c['perplexity_samples']:.0f}). "
          "A sampled perplexity much lower than the training perplexity means the prior concentrates on a subset of codes (mode drop).\n")
    w(F("fig10_diversity.png", "Diversity: mode occupancy, pairwise SSIM and code usage"))

    w("## 4. Quantitative benchmark\n")
    w(_tbl(["Metric", *[L[k] for k in models]],
           [["Test reconstruction SSIM, mean [95% CI] (paired; target > 0.6)", *[_ci_text(rec[m]["ssim"]) for m in models]],
            ["Share of test slices with SSIM > 0.6", *[f"{rec[m]['fraction_above_target']:.1%}" for m in models]],
            ["Reconstruction MSE", *[f"{rec[m]['mse_mean']:.5f}" for m in models]],
            ["Synthetic best-match SSIM to real train slices, mean [95% CI]", *[_ci_text(mem[m]["best_match_ssim"]) for m in models]],
            ["Share of synthetic images with best-match SSIM > 0.6", *[f"{mem[m]['fraction_above_target']:.1%}" for m in models]]]) + "\n")
    w(f"Real-test floor for best-match SSIM: {_ci_text(mem['real_test']['best_match_ssim'])}. "
      "Reconstruction SSIM is a paired comparison and measures the autoencoder; best-match SSIM compares *unpaired* synthetic images with their most similar real slice and measures generated-cohort fidelity.\n")
    w(_tbl([f"Distribution fidelity ({fid['feature_name']})", *[L[k] for k in ["real_test", *models]]],
           [["Frechet distance to real train (lower better)", *[f"{fid['fd'][k]:.2f}" for k in ["real_test", *models]]],
            ["Frechet distance, pixel-PCA space", *[f"{fid['fd_pixel_pca'][k]:.2f}" for k in ["real_test", *models]]],
            *[[n.capitalize() + " (k=5)", *[f"{fid['prdc'][k][n]:.3f}" for k in ["real_test", *models]]] for n in ("precision", "recall", "density", "coverage")]]) + "\n")
    w(f"{fid['note']} Precision/recall/density/coverage use equal-size subsets (n = {fid['prdc_n_per_set']} per set).\n")
    w(F("fig09_metric_comparison.png", "SSIM, Frechet distance and precision / recall / density / coverage vs the real-test floor"))
    w(F("fig04_reconstructions.png", "Evenly spaced test reconstructions with error maps"))
    w("### Resource profile\n")
    w(_tbl(["", *[L[m] for m in models]],
           [["Parameters, autoencoder (M)", *[f"{res[m]['autoencoder_params_m']:.2f}" for m in models]],
            ["Parameters, prior (M)", *[f"{res[m]['prior_params_m']:.2f}" for m in models]],
            ["Training wall time incl. prior (min)", *[f"{res[m]['train_minutes']:.1f}" if res[m]["train_minutes"] else "n/a" for m in models]],
            ["Peak accelerator memory, training (MB)", *[f"{res[m]['train_peak_mb']:.0f}" if res[m]["train_peak_mb"] else "n/a" for m in models]],
            ["Generation latency (ms / sample, incl. decode)", *[f"{res[m]['gen_ms_per_sample']:.1f}" for m in models]],
            ["Peak accelerator memory, generation (MB)", *[f"{res[m]['gen_peak_mb']:.0f}" if res[m]["gen_peak_mb"] else "n/a" for m in models]]]) + "\n")
    w(F("fig15_resource_profile.png", "Parameters, training time, memory and generation latency"))

    w("## 5. Qualitative autopsy and memorisation audit\n")
    w("### 5.1 Uncurated sample grids\n")
    for m in models:
        w(F(f"fig05_samples_{m}.png", f"{L[m]}: the first 32 samples for each random seed (nothing selected or discarded)"))
    w("### 5.2 t-SNE embedding of real vs synthetic\n")
    w(F("fig13_embedding.png", f"{R['embedding_method']} of {fid['feature_name']}. Synthetic points that sit in empty regions are off-manifold; real regions with no synthetic points are mode drops"))
    w("### 5.3 Failure autopsy (3 to 5 flawed generations, selected automatically)\n")
    w("To avoid cherry-picking, each failure is the single *worst* sample under one objective detector (Nyquist-corner energy, body-mask fragmentation, "
      "feature-space distance from real data, lowest Laplacian sharpness, lowest best-match SSIM). **Reviewer diagnosis cells must be completed by a human after viewing the figure.**\n")
    for m in models:
        w(f"**{L[m]}**\n")
        w(_tbl(["Failure type", "Seed / index", "Automatic evidence", "Nearest train case (SSIM)", "Reviewer diagnosis (fill in)"],
               [[a["failure_type"], f"{a['seed']} / {a['index']}", a["evidence"], f"{a['nearest_case']} ({a['nearest_ssim']:.2f})", "_to be completed_"] for a in R["autopsy"][m]]) + "\n")
        w(F(f"fig12_failure_autopsy_{m}.png", f"{L[m]} failure autopsy: generated | nearest training slice | difference"))
    w("### 5.4 Nearest-training-neighbour audit\n")
    keys = [k for k in ["real_test", *models, "control"] if k in mem]
    w(_tbl(["", *[L[k] for k in keys]],
           [["Mean nearest-train pixel MSE", *[f"{mem[k]['d1_mean']:.5f}" for k in keys]],
            [f"Close-match rate (<= real-test 5th percentile, MSE {mem['close_match_threshold_mse']:.5f})", *[f"{mem[k]['flagged_close_match_rate']:.1%}" for k in keys]],
            [f"Near-duplicate rate (best-match SSIM >= {NEAR_DUP_SSIM})", *[f"{mem[k]['near_duplicate_rate']:.1%}" for k in keys]],
            ["Highest best-match SSIM", *[f"{mem[k]['best_match_ssim_max']:.3f}" for k in keys]],
            ["Distinct training cases among nearest matches", *[str(mem[k]["distinct_nearest_cases"]) for k in keys]],
            ["Share of matches pointing at the single most-matched case", *[f"{mem[k]['top_case_share']:.1%}" for k in keys]]]) + "\n")
    w("The *real test* column shows what an unseen-patient cohort looks like (close-match rate 5% by construction). The *train reconstructions* column is a **positive control**: "
      "images that genuinely are copies of training slices. If the audit cannot separate that column from the synthetic ones it has no power; a clear gap means a copy-like cohort would have been detected.\n")
    w(_tbl(["Data-copying test (equal-size train subset vs test set)", *[L[m] for m in models]],
           [["Fraction of synthetic images closer to TRAIN than TEST (0.5 = no copying)", *[f"{mem[m]['data_copying_test']['fraction_closer_to_train']:.3f}" for m in models]],
            ["Two-sided binomial p-value", *[f"{mem[m]['data_copying_test']['p_value_two_sided']:.3g}" for m in models]]]) + "\n")
    w(F("fig07_nearest_neighbour_audit.png", "Nearest-neighbour distance, best-match SSIM and data-copying test"))
    for m in models:
        w(F(f"fig08_closest_to_training_{m}.png", f"{L[m]}: the 6 synthetic images closest to a training slice, with their matches"))
    w("### 5.5 Latent interpolation between synthetic images and training scans\n")
    w(_tbl(["Interpolation path smoothness", *[L[m] for m in models]],
           [["Mean jump ratio (max step / mean step; 1 = perfectly even)", *[f"{R['interpolation'][m]['mean_jump_ratio']:.2f}" if R["interpolation"][m]["mean_jump_ratio"] else "n/a" for m in models]],
            ["Worst jump ratio", *[f"{R['interpolation'][m]['max_jump_ratio']:.2f}" if R["interpolation"][m]["max_jump_ratio"] else "n/a" for m in models]]]) + "\n")
    w(F("fig14_latent_interpolation.png", "Linear latent interpolation from a synthetic sample to its nearest training slice (top rows per model) and to a random training slice (bottom rows)"))
    w("Smooth morphing with no abrupt jumps indicates the latent space generalises between patients rather than storing isolated training examples; a sudden snap to the endpoint near alpha = 1 would indicate a memorised basin.\n")

    w("## 6. Engineering recommendation\n")
    for m in models:
        v = R["verdict"][m]
        w(f"- **{L[m]}**: {v['label']}." + "".join(f" Failed ({g.lower()}): " + "; ".join(c) + "." for g, c in v["failed"].items() if c))
    w("\n**What this evidence supports.** Under the criteria above, the audit looks for generated images that are closer to training slices than real unseen-patient slices are, "
      "for cohort-level copying, and for loss of diversity or realism relative to the real-data floor.\n")
    w("**What it does not establish.**\n"
      "- Absence of memorisation is *not* privacy. Pixel / feature nearest-neighbour distances miss memorised attributes that are not visually obvious; membership-inference attacks and, where required, differential-privacy training are the stronger tools.\n"
      "- SSIM, FD and precision/recall are proxies. None validates clinical plausibility; a radiologist review of the failure autopsy and a downstream-task test (train on synthetic, test on real) are still required before the cohort is used for diagnostic models.\n"
      "- Slices come from a small number of patients; adjacent slices of one patient are near-identical, so the audit groups by case ID, but patient-level leakage can still occur between visits that share anatomy.\n"
      "- Metrics depend on the sample size and feature extractor (see Section 1); the real-test floor is included for that reason.\n")
    w("**Suggested release gate:** share the cohort only if (i) all memorisation criteria pass, (ii) a clinician has reviewed Section 5.3, and (iii) a downstream model trained on the synthetic data performs within an agreed margin of one trained on real data.\n")
    path.write_text("\n".join(out), encoding="utf-8")

