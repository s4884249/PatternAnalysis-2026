# Generating Synthetic HipMRI Slices with a VQ-VAE and Autoregressive Prior

**Difficulty:** Hard

**Student ID:** s4884249

## 1. Project Overview

Medical-imaging research is limited by patient-data privacy requirements and the availability of clinical scans. Synthetic images could support research and data augmentation, but only if they retain useful anatomical structure, provide diversity, and do not reproduce identifiable training examples.

This project develops a **Vector-Quantised Variational Autoencoder (VQ-VAE)** to generate synthetic 2D HipMRI slices from `keras_slices_data`. A **Variational Autoencoder (VAE)** is the generative baseline. The investigation asks whether discrete spatial representations and an autoregressive prior improve image fidelity and distribution coverage without causing codebook collapse, implausible anatomy, or obvious training-image memorisation. The intended users are medical-imaging researchers and machine-learning engineers evaluating synthetic scans for research use; the outputs are not intended for clinical use.

![Final VAE and VQ-VAE architecture](images/final_architecture.png)

_Figure 1: Baseline VAE and selected VQ-VAE architecture._

## 2. Feasibility Review

### Scope and acceptance criteria

The prototype is limited to 2D HipMRI generation. Both models use the same patient-level data partitions and evaluation procedures.

- [x] Train a VQ-VAE that reconstructs HipMRI slices.
- [x] Generate new samples by sampling discrete latent codes from a learned prior.
- [x] Exceed the SSIM target of `0.6` on held-out paired reconstructions.
- [x] Compare the VQ-VAE with a simpler generative baseline using the same data splits.
- [x] Inspect uncurated samples, latent interpolations, and nearest training neighbours for diversity, anatomical plausibility, and possible memorisation.
- [x] Measure parameter count, training time, peak accelerator memory, and generation latency.

### Preliminary evidence

The initial Rangpur audit inspected `25,320` NIfTI files. All were readable `float32` files with finite voxel values. The raw dimensions were `256 × 128` for `25,200` files and `256 × 144` for `120` files; the observed intensity range was `[0, 1212]`. The directory contained both MRI images and segmentation masks, so image/mask classification and patient-level grouping were required before training.

A small deterministic autoencoder smoke test verified loading, forward and backward passes, finite losses, GPU execution, reconstruction, and checkpoint save/load. It preserved large anatomical features but blurred fine detail. This established that the pipeline worked; it did not establish generative ability, because a deterministic autoencoder does not learn a sampling distribution. The final baseline was therefore upgraded to a VAE.

The initial compute plan reserved **6–12 GPU hours on Rangpur** for data preparation, smoke tests, experiments, and final evaluation. This was a planning estimate, not a measured runtime. The final measured training times are reported in Section 7.

### Main risks and fallback

| Risk | Mitigation or fallback |
|---|---|
| Poor reconstruction or unstable training | Inspect validation curves and image grids; verify scaling and losses; reduce resolution, channel widths, or epochs. |
| Codebook collapse | Monitor active entries and perplexity; adjust codebook size and optimisation. |
| Training-image memorisation | Keep patient splits isolated and audit nearest training neighbours. |
| Mode collapse or limited diversity | Inspect uncurated samples across multiple random seeds and compare feature-space coverage. |
| Excessive memory or runtime | Start with compact models, profile each execution stage, and reduce model capacity if required. |
| Unreliable comparisons | Use the same manifest, patient partitions, and held-out test set; report reconstruction and generation metrics separately. |

If quantisation remains unstable, the VAE remains a functioning generative baseline for diagnosing the failure. Poor samples are not treated as successful merely because the model produces images; the final recommendation depends on fidelity, diversity, memorisation checks, and measured resource use.

## 3. Dataset and Preprocessing

The dataset contains HipMRI 2D slices and corresponding segmentation masks. Generation uses the MRI images; the masks are retained for audit and pairing checks, not as model targets. The manifest contains `12,660` image-mask pairs across `38` patient cases. Its partitions are `11,460` training, `660` validation, and `540` test images, with `31`, `4`, and `3` patients respectively. No patient appears in more than one split.

The loader keeps the standard `256 × 128` geometry and excludes the `60` MRI slices with `256 × 144` geometry. This leaves **12,600 model images** with a consistent aspect ratio. The final pipeline downsamples these slices to `1 × 128 × 64` model inputs while preserving the source aspect ratio.

![Representative HipMRI input slices from the data partitions](images/explore_sample_slices.png)

_Figure 2: Representative slices from the training, validation, and test partitions._

### Preprocessing

Each slice is clipped to its 1st and 99.5th intensity percentiles, then scaled to `[0, 1]`:

```python
low, high = np.percentile(array, [1.0, 99.5])
if high <= low:
    return np.zeros_like(array, dtype=np.float32)
return ((np.clip(array, low, high) - low) / (high - low)).astype(np.float32)
```

The manifest loader excludes masks, unreadable files, non-standard shapes, constant slices, and (when configured) slices below a minimum non-zero-pixel fraction. Per-slice normalisation reduces the influence of extreme intensities and between-scan scale variation, but removes absolute signal information. Generated intensities therefore must not be interpreted as quantitative MRI measurements.

Patient-level splitting is essential because slices from the same patient are correlated. A random slice-level split could place near-duplicate anatomy in training and test partitions and inflate results. The visual partition check shows broadly similar anatomical content, but does not prove statistical equivalence between the patient groups. The small number of held-out patients also limits generalisation claims.

## 4. Model Design

**Selected model:** VQ-VAE with a causal-transformer prior  
**Difficulty:** Hard  
**Generative baseline:** VAE

### VAE baseline

The VAE encoder uses three stride-2 convolutional stages with Group Normalisation and SiLU, reducing a `1 × 128 × 64` slice to a feature map that is projected to a 256-dimensional Gaussian latent distribution. Training samples from this distribution using the reparameterisation trick; reconstruction at evaluation uses the latent mean. Its objective is:

```text
L_VAE = L_MSE + 0.01 * D_KL / 8192
```

For generation, a full-covariance Gaussian is fitted to training-set latent vectors and sampled before a single decoder pass. This provides a more useful sampling distribution than the unregularised latent space of the initial deterministic autoencoder.

### VQ-VAE autoencoder and codebook

The VQ-VAE encoder produces a `64 × 32 × 16` latent tensor: a `32 × 16` spatial grid with 64 features per position. Each of the 512 vectors is replaced by its nearest entry in a learned codebook of 512 64-dimensional vectors. A straight-through estimator passes decoder gradients to the encoder, while exponential moving averages (EMA, decay `0.99`) update codebook entries. Data-dependent initialisation and inactive-code restarts reduce dead-code problems.

The decoder maps the quantised grid back to a `1 × 128 × 64` image. Residual blocks support feature learning, and nearest-neighbour upsampling followed by convolution avoids the uneven-overlap patterns associated with transposed-convolution upsampling. The reconstruction objective is:

```text
L_VQ = L_MSE + 0.1 * (1 - SSIM) + 0.25 * L_commitment
```

MSE penalises pixel differences, SSIM encourages local structural similarity, and commitment loss keeps encoder outputs close to selected codebook vectors. The codebook is EMA-updated rather than optimiser-trained, so its `32,768` values are excluded from the trainable parameter count.

### Autoregressive prior

The `32 × 16` code grid is flattened in raster order into 512 tokens. A six-block causal transformer with eight attention heads, hidden dimension 256, and a `256 → 1024 → 256` feed-forward network predicts each code from preceding codes. Learned row and column embeddings preserve spatial position. The prior is trained after the autoencoder using next-token cross-entropy, AdamW, warm-up, cosine learning-rate decay, and early stopping on validation cross-entropy.

Generation begins with a beginning-of-sequence token and predicts 512 code indices sequentially. A key/value cache avoids recomputing prior attention states. The sampled code vectors are then decoded in one pass. The sequential prior is the main source of the VQ-VAE's higher generation latency.

### Development decisions

- The initial deterministic autoencoder was suitable for a reconstruction smoke test but not reliable prior sampling. It was replaced by a VAE baseline with a fitted aggregate latent Gaussian.
- Early VAE prior samples were mostly texture. The final VAE uses a 256-dimensional global latent representation and a full-covariance Gaussian fitted to training latents.
- Early VQ-VAE samples had locally plausible texture but inconsistent global anatomy. The final model increased the code grid to `32 × 16`, expanded the codebook to 512 entries, added residual blocks and a six-layer causal transformer, and used validation-based early stopping.
- The final VQ-VAE reconstruction loss includes SSIM with weight `0.1`; this improved local structure in the sample grids but does not establish anatomical validity.

## 5. Installation, Reproducibility, and Usage

### Environment and dependencies

The recorded environment uses the following versions:

| Dependency | Version |
|---|---:|
| Python | `3.14.7` |
| NumPy | `2.5.4` |
| PyTorch | `2.14.1` |
| nibabel | `5.4.2` |
| scikit-image | `0.26.0` |
| scikit-learn | `1.9.1` |
| Matplotlib | `3.11.2` |
| torchvision | `0.29.2` |

Install the requirements in a virtual environment. The following commands are for macOS with Homebrew:

```bash
brew install python@3.14
/opt/homebrew/bin/python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`device_utils.py` selects CUDA, Apple MPS, or CPU where available. Model settings, defaults, and run-directory naming are centralised in `params.py`.

### Run commands

Run these commands from `recognition/VQ-VAE_s4884249/`:

```bash
# Audit the NIfTI files and create a manifest
python dataset.py audit --data-root /path/to/keras_slices_data --output manifest.csv

# Inspect the dataset and create exploratory figures
python explore.py --data-root /path/to/keras_slices_data --manifest manifest.csv

# Train the VAE baseline and selected VQ-VAE
python train.py --model vae --data-root /path/to/keras_slices_data
python train.py --model vqvae --data-root /path/to/keras_slices_data --ssim-weight 0.1

# Reconstruct slices and generate samples from saved checkpoints
python predict.py --checkpoint runs/vae_seedx42-epochsx50 --top-k 50
python predict.py --checkpoint runs/vqvae_seedx42-epochsx50-ssim-weightx0.1 --top-k 50

# Compare both runs and write evaluation outputs
python evaluate.py \
    --vae-run runs/vae_seedx42-epochsx50 \
    --vqvae-run runs/vqvae_seedx42-epochsx50-ssim-weightx0.1 \
    --data-root /path/to/keras_slices_data --out evaluation
```

**Example input:** a NIfTI HipMRI slice from `keras_slices_data`, read through the manifest and normalised to `[0, 1]`.  
**Example outputs:** a reconstructed slice, prior-generated synthetic slices, sample grids, nearest-neighbour comparisons, interpolation visualisations, and evaluation metrics. The figures in Sections 6 and 7 show representative inputs, reconstructions, and generated outputs.

### Main project files

| File | Responsibility |
|---|---|
| `modules.py` | VAE, residual VQ-VAE encoder/decoder, EMA quantiser, and causal-transformer prior. |
| `dataset.py` | NIfTI audit, manifest creation, preprocessing, patient-level splits, and leakage checks. |
| `train.py` | Training, validation, checkpoint selection, test metrics, and resource logging. |
| `predict.py` | Checkpoint inference, sample generation, reconstruction, interpolation, and nearest-neighbour visualisations. |
| `evaluate.py` | Side-by-side VAE/VQ-VAE evaluation and cohort-level analysis. |
| `eval_metrics.py`, `eval_plots.py`, `eval_analysis.py` | Distribution metrics, diagnostic plots, and pass/fail analysis. |
| `params.py`, `utils.py`, `device_utils.py` | Configuration, shared helpers, and accelerator selection/profiling. |
| `requirements.txt` | Pinned Python dependencies. |

Datasets and model checkpoints are not part of the source submission; paths to local data and saved runs must be supplied when running the commands.

## 6. Experimental Results

### Evaluation protocol

The final comparison uses the same patient-level manifest and random seed `42`; both autoencoders were trained for 50 epochs. The VQ-VAE prior was trained separately and selected using validation cross-entropy. The test partition was held out for final evaluation. Reconstruction SSIM measures paired input/reconstruction similarity; Fréchet distance and feature-space measures compare generated and real cohorts; nearest-neighbour analysis investigates possible copying. These metrics address different properties and should not be treated as interchangeable.

### Training and reconstruction

![Training curves for the VAE and VQ-VAE](images/evaluation_training_curves.png)

_Figure 3: Training losses, validation SSIM, reconstruction MSE, gradient norms, codebook use, and learning rate._

The VQ-VAE achieved higher validation SSIM and lower reconstruction MSE. Its gradient norms declined during training, and nearly all codebook entries were used. Total loss values are not directly comparable because the VAE includes KL regularisation while the VQ-VAE includes codebook and commitment terms.

![Transformer prior training curves](images/evaluation_prior_training.png)

_Figure 4: Transformer-prior next-code cross-entropy._

Validation cross-entropy plateaued at approximately epoch 10, supporting early stopping rather than selecting the final training epoch. Uniform prediction over 512 code entries corresponds to approximately `6.24` nats per code; both training and validation losses fell below this reference, showing that the prior learned code dependencies.

![Held-out test reconstructions and error maps](images/evaluation_reconstructions.png)

_Figure 5: VAE and VQ-VAE test reconstructions with error maps._

The VAE preserves large structures but blurs fine boundaries. VQ-VAE reconstructions retain more detail and have lower errors, consistent with their higher SSIM.

### Generated samples and quantitative comparison

![VAE generated samples](images/evalutaion_samples_vae.png)

_Figure 6: Uncurated VAE samples._

![VQ-VAE generated samples](images/evalutaion_samples_vqvae.png)

_Figure 7: Uncurated VQ-VAE samples._

The VAE produces recognisable MRI-like images but often blurs tissue boundaries. VQ-VAE samples generally retain sharper local patterns, although some samples contain irregularities and anatomically inconsistent structures. Uncurated grids and diagnostic samples from multiple random seeds were inspected to avoid relying on a single favourable sample.

| Metric | VAE baseline | VQ-VAE | Held-out real reference |
|---|---:|---:|---:|
| Test reconstruction SSIM | `0.713` | `0.950` | — |
| Best-match SSIM to training images | `0.293` | `0.341` | `0.343` |
| Fréchet distance (Inception-v3 features; lower is better) | `~226` | `75.14` | `51.59` |

Both models exceed the `0.6` target on **paired test reconstructions**. The VQ-VAE also has a substantially lower Fréchet distance than the VAE, although its distance remains higher than the real-versus-real reference. Precision, recall, density, and coverage also favoured the VQ-VAE, particularly recall, but its precision, density, and coverage remained below the held-out real reference.

> **Metric distinction:** The `0.6` SSIM target is met by the paired reconstruction results. The best-match SSIM values in the second row are nearest-neighbour similarity measurements, not the same metric or a pass/fail test for generated-image fidelity.

### Distribution and codebook analysis

![t-SNE comparison of real and generated feature distributions](images/evaluation_embedding.png)

_Figure 8: t-SNE of Inception-v3 features for training, held-out real, VAE-generated, and VQ-VAE-generated images._

The VAE samples form a more separated feature cluster. VQ-VAE samples lie closer to the real-data groups, but some separation remains. t-SNE is qualitative: a two-dimensional projection can distort distances and does not prove that the distributions are equivalent.

![Codebook utilisation](images/codebook_utilisation.png)

_Figure 9: Codebook assignment distribution._

| Codebook metric | Training images | Held-out test images | Prior-generated samples |
|---|---:|---:|---:|
| Active entries | `512/512` | `509/512` | `511/512` |
| Perplexity | `389` | `369` | `384` |
| Codes accounting for 90% of assignments | `359` | `341` | `360` |

Use is non-uniform, but activity across nearly all 512 entries and the reported perplexities provide no evidence of severe codebook collapse. The prior also broadly reproduces the marginal code-frequency distribution.

### Nearest-neighbour and memorisation audit

![Nearest-neighbour audit](images/evaluation_nearest_neighbour_audit.png)

_Figure 10: Generated-sample distances and similarities compared with training and held-out real images._

VQ-VAE mean best-match SSIM to training images was `0.341`, close to `0.343` for held-out real images. Nearest-neighbour examples show differences in tissue patterns, boundaries, and intensities rather than exact copies. The train-versus-test distance plot shows a slight tendency for generated samples to be closer to a training neighbour; it does not show obvious systematic pixel-level copying. These are checks for visible similarity, not a formal privacy guarantee.

![VQ-VAE generated samples and their nearest training neighbours](images/evaluation_closest_to_training_vqvae.png)

_Figure 11: VQ-VAE samples, nearest training slices, and difference maps._

### Failure-case autopsy

The following five VQ-VAE samples were selected automatically using diagnostic criteria, not hand-picked examples. Each is compared with its nearest training slice in the failure figure. The metrics identify observable image defects; they are not clinical diagnoses.

![Automatically selected VQ-VAE failure cases and nearest training slices](images/evaluation_failure_autopsy_vqvae.png)

_Figure 12: Five automatically selected VQ-VAE generation failures, nearest training neighbours, and difference maps._

| Failure mode | Generated sample | Diagnostic evidence | Nearest training case | SSIM |
|---|---|---|---|---|
| High-frequency / checkerboard artefact | Seed 2, sample 25 | Nyquist-corner energy `0.00008`, above the real-data 95th percentile (`0.000025`) | `case_015` | `0.37` |
| Fragmented tissue margin | Seed 1, sample 15 | Body-mask fragmentation `0.74` across 21 components; real-data 95th-percentile fragmentation was `0.76` | `case_019` | `0.22` |
| Off-manifold anatomy | Seed 4, sample 8 | Mean 5-NN feature distance `14.2`, versus cohort median `10.6` | `case_012` | `0.25` |
| Blur / lost detail | Seed 1, sample 17 | Laplacian variance `0.00605`, below the real-data median (`0.02`) | `case_027` | `0.46` |
| No close anatomical match | Seed 1, sample 3 | Best SSIM to any training slice was `0.18` | `case_023` | `0.18` |

The cases expose different failure mechanisms: high-frequency artefacts, disconnected body regions, feature-space outliers, loss of fine detail, and outputs with no close training analogue. Nearest-neighbour similarity alone cannot determine whether an outlier is novel or anatomically invalid, so the images require expert review before any clinical interpretation.

### Computational profile

Computational profiling was performed on an Apple M5 Pro (64 GiB unified memory) using PyTorch 2.14.1 and the MPS backend, with float32 inputs of size 128 × 64. Measurements use warmed-up runs and report median latency; generation was measured over five runs. The models were randomly initialised, so these results characterise architectural resource requirements rather than the performance of trained checkpoints.

| Metric | VAE | VQ-VAE |
|---|---:|---:|
| Trainable parameters | 3.55M | 7.02M |
| Resident weights | 13.6 MiB | 27.0 MiB |
| Peak reconstruction memory (batch 64) | 40.7 MiB | 226.1 MiB |
| Peak training-step memory (batch 64) | 229.1 MiB | 2,285.5 MiB |
| Training-step latency (batch 64) | 23.8 ms | 172.7 ms |
| Generation latency (batch 1) | 0.44 ms | 636.9 ms |
| Generation throughput (batch 64) | 15,780.6 images/s | 32.0 images/s |

The VAE was substantially more computationally efficient, requiring approximately one-tenth of the VQ-VAE's peak training-step memory and completing a batch-64 training step around 7.3 times faster. The VQ-VAE's autoregressive transformer prior is the main computational bottleneck: sampling 512 latent codes takes approximately 663 ms per image, compared with 0.99 ms for the decoder alone. Prior training is particularly memory-intensive, reaching approximately 13 GiB at batch size 64.

Both models completed the measured operations, but these benchmarks do not establish training convergence or numerical stability. The higher VQ-VAE training latency and memory requirements make it more sensitive to batch size and available unified memory. Its autoregressive generation also limits practical sampling throughput. These costs should be considered when selecting batch sizes, planning experiments, and generating synthetic datasets. MPS memory measurements are approximate, include allocator caching where reported by the driver, and may vary with system memory pressure and thermal conditions.

## 7. Hyperparameter Experiments and Further Actions

### Experimental protocol

Three one-factor-at-a-time hyperparameter sweeps were conducted using `experiments.py`. Each configuration was trained for 30 epochs across three random seeds (42, 43, 44), with results reported as mean ± standard deviation. Unless otherwise specified, the VQ-VAE configuration used the default codebook size of 512, an SSIM loss weight of 0, and two downsampling stages. The SSIM loss weight of 0 was used rather than the current model's 0.1 for a closer comparison with the baseline model.

Reconstruction quality was assessed using test SSIM and mean squared error (MSE). Generated-sample quality was assessed using best-match SSIM, pixel-PCA Fréchet distance (FD), precision, recall, coverage, close-match rate, and sharpness ratio. Codebook utilization and prior validation cross-entropy (CE) were recorded where applicable.

### 1. Codebook size

| Codebook size | Test SSIM ↑ | Test MSE ↓ | FD ↓ | Precision ↑ | Recall ↑ | Best-match SSIM ↑ | Close-match rate ↓ | Sharpness ratio | Perplexity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.8882 ± 0.0013 | 0.001485 ± 0.000017 | 1.098 ± 0.340 | 0.9674 ± 0.012 | 0.8711 ± 0.043 | 0.3465 ± 0.015 | 0.1484 ± 0.032 | 0.528 ± 0.032 | 51.63 ± 0.99 |
| 128 | 0.9013 ± 0.0031 | 0.001302 ± 0.000031 | 1.011 ± 0.170 | 0.9648 ± 0.035 | **0.9023 ± 0.014** | **0.3599 ± 0.010** | **0.1341 ± 0.037** | 0.5735 ± 0.038 | 101.3 ± 3.3 |
| 256 | 0.9140 ± 0.0013 | 0.001144 ± 0.000020 | 1.062 ± 0.220 | 0.9714 ± 0.036 | 0.8620 ± 0.066 | 0.3551 ± 0.013 | 0.1419 ± 0.063 | 0.5733 ± 0.023 | 192.4 ± 0.73 |
| **512 (defailt)** | 0.9245 ± 0.0034 | 0.001016 ± 0.000043 | 0.8993 ± 0.060 | 0.9701 ± 0.025 | 0.8789 ± 0.031 | 0.3540 ± 0.0072 | 0.2096 ± 0.044 | 0.6035 ± 0.063 | 360.1 ± 5.5 |
| 1024 | **0.9323 ± 0.0033** | **0.0009107 ± 0.000039** | **0.8777 ± 0.013** | **0.9753 ± 0.0081** | 0.8581 ± 0.046 | 0.3519 ± 0.0014 | 0.1445 ± 0.028 | **0.655 ± 0.030** | 721.2 ± 6.2 |

*Arrows indicate the preferred direction. Values are mean ± standard deviation across three seeds.*

Increasing the codebook size improved reconstruction quality monotonically. Increasing it from 512 to 1024 raised test SSIM from 0.9245 to 0.9323 and reduced MSE from 0.001016 to 0.0009107. FD also decreased from 0.8993 to 0.8777, while precision and sharpness ratio increased.

However, these gains did not extend to every criterion. Recall and best-match SSIM were highest at 128, and the close-match rate at 512 was substantially higher than at the other tested sizes. The 1024-entry codebook therefore offers a promising reconstruction and sample-quality trade-off, but its larger latent vocabulary does not by itself establish better privacy or anatomical validity. The differences in FD and close-match rate should also be interpreted in light of variation across seeds.

### 2. SSIM loss weight

| Model | SSIM weight | Test SSIM ↑ | Test MSE ↓ | FD ↓ | Precision ↑ | Recall ↑ | Best-match SSIM ↑ | Close-match rate ↓ | Sharpness ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VAE | 0 | 0.6970 ± 0.0008 | **0.004430 ± 0.000033** | **0.5176 ± 0.046** | 0.7214 ± 0.013 | 0.8477 ± 0.027 | 0.2895 ± 0.0028 | 0.0599 ± 0.016 | 0.2864 ± 0.0073 |
| VAE | 0.05 | 0.7403 ± 0.0006 | 0.004643 ± 0.000014 | 0.5267 ± 0.052 | 0.6680 ± 0.020 | 0.8698 ± 0.009 | 0.2846 ± 0.0020 | **0.0326 ± 0.011** | 0.3524 ± 0.0060 |
| VAE | 0.1 | 0.7444 ± 0.0008 | 0.004847 ± 0.000012 | 0.5474 ± 0.070 | 0.6641 ± 0.0068 | 0.8750 ± 0.0068 | 0.2832 ± 0.0011 | 0.0391 ± 0.010 | 0.3727 ± 0.0090 |
| VAE | 0.25 | 0.7463 ± 0.0007 | 0.005097 ± 0.000026 | 0.5647 ± 0.071 | 0.6510 ± 0.032 | 0.8607 ± 0.015 | 0.2829 ± 0.0030 | 0.0404 ± 0.023 | **0.3895 ± 0.0032** |
| VAE | 0.5 | 0.7462 ± 0.0008 | 0.005285 ± 0.000028 | 0.5991 ± 0.089 | 0.6458 ± 0.034 | **0.8828 ± 0.0068** | 0.2832 ± 0.0035 | 0.0391 ± 0.014 | 0.3852 ± 0.013 |
| VQ-VAE | **0 (default)** | 0.9245 ± 0.0034 | 0.001016 ± 0.000043 | **0.8993 ± 0.060** | 0.9701 ± 0.025 | 0.8789 ± 0.031 | 0.3540 ± 0.0072 | 0.2083 ± 0.043 | 0.6035 ± 0.063 |
| VQ-VAE | **0.05** | 0.9438 ± 0.0013 | **0.0009531 ± 0.000022** | 0.9916 ± 0.240 | 0.9479 ± 0.018 | 0.8867 ± 0.017 | 0.3528 ± 0.0063 | **0.0898 ± 0.047** | 0.7522 ± 0.043 |
| VQ-VAE | **0.1 (current)** | 0.9458 ± 0.0010 | 0.0009610 ± 0.000020 | 1.082 ± 0.390 | 0.9518 ± 0.0045 | 0.8945 ± 0.050 | **0.3583 ± 0.0054** | 0.0977 ± 0.038 | **0.7633 ± 0.034** |
| VQ-VAE | 0.25 | 0.9477 ± 0.0005 | 0.0009792 ± 0.000011 | 0.8924 ± 0.210 | **0.9635 ± 0.018** | **0.8984 ± 0.048** | 0.3510 ± 0.017 | 0.1263 ± 0.033 | 0.7490 ± 0.040 |
| VQ-VAE | 0.5 | **0.9481 ± 0.0005** | 0.001010 ± 0.000010 | 1.044 ± 0.260 | 0.9753 ± 0.011 | 0.8828 ± 0.027 | 0.3395 ± 0.018 | 0.1055 ± 0.014 | 0.7180 ± 0.033 |

*Values are mean ± standard deviation. Bold identifies selected reference values or the best observed value for a particular metric within each model; it does not imply statistical significance.*

For the VAE baseline, increasing the SSIM weight improved test SSIM but increased MSE and FD. A weight of 0.05 produced the lowest close-match rate, while larger weights generally increased the sharpness ratio. This illustrates that better reconstruction structure does not necessarily imply better generated-sample distribution quality.

For the VQ-VAE, a weight of 0.05 is a promising alternative to the current weight of 0.1. It achieved a test SSIM of 0.9438, a slightly lower MSE of 0.0009531, and a lower close-match rate of 0.0898, compared with 0.9458, 0.0009610, and 0.0977 at 0.1. Its recall was also slightly lower, while FD was lower than at 0.1 but remained variable across seeds.

A weight of 0.25 produced the highest mean recall in this sweep, whereas 0.5 produced the highest mean test SSIM and precision. Neither was uniformly better: best-match SSIM declined as the weight increased beyond 0.1, and the close-match rate increased at 0.25 and 0.5. The modest reconstruction gains at higher weights therefore do not justify selecting them on SSIM alone.

### 3. Number of downsampling stages

| Downsampling stages | Parameters | Test SSIM ↑ | Test MSE ↓ | FD ↓ | Precision ↑ | Recall ↑ | Best-match SSIM ↑ | Close-match rate ↓ | Sharpness ratio | Training time (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **2 (default)** | 2.003M | **0.9245 ± 0.0034** | **0.001016 ± 0.000043** | **0.8993 ± 0.060** | **0.9701 ± 0.025** | **0.8789 ± 0.031** | **0.3540 ± 0.0072** | 0.2083 ± 0.043 | **0.6035 ± 0.063** | 4773 ± 340 |
| 3 | 7.924M | 0.7754 ± 0.0076 | 0.003348 ± 0.000100 | 1.067 ± 0.150 | 0.9570 ± 0.017 | 0.8346 ± 0.012 | 0.3538 ± 0.0055 | **0.1549 ± 0.047** | 0.4142 ± 0.027 | 5199 ± 290 |

Two downsampling stages substantially outperformed three in reconstruction quality, FD, precision, recall, and sharpness ratio, while requiring fewer parameters and less training time in this sweep. Three stages reduced the close-match rate, but the result was accompanied by substantially worse reconstruction and distribution metrics. The similar best-match SSIM values do not offset these differences.

The two-stage architecture should therefore remain the default. A deeper encoder-decoder is not justified by these results, although the lower close-match rate at three stages warrants further investigation rather than being interpreted as conclusive evidence of improved privacy.

### 4. Interpretation and limitations

The sweeps reveal trade-offs between reconstruction fidelity, sample diversity, distributional similarity, and potential memorisation. Although all configurations exceeded the reconstruction SSIM threshold, none provided uniformly improved results that would justify an imediate change to the model configuration.

The held-out real-data reference values were FD = 3.7, precision = 0.875, recall = 0.0703, coverage = 0.492, density = 0.777, best-match SSIM = 0.335, and pairwise SSIM = 0.164. The generated-sample FD values were lower than the real-data reference under the same evaluation procedure, but this should not be interpreted as proof that synthetic samples are more realistic. These metrics use pixel-PCA features and small cohorts, so their absolute values are not equivalent to published FID results. In particular, the very different precision and recall values indicate that the generated and real sample distributions behave differently under the evaluator.

Close-match rate is a useful screening measure, not a privacy guarantee. Best-match SSIM also measures similarity rather than clinical correctness: a high value can indicate faithful anatomical structure, but it can also indicate excessive similarity to a training example. Conversely, a low value does not establish anatomical plausibility or safety. None of the reported metrics establishes concrete diagnostic utility, preservation of pathology, or absence of patient information.

Differences should be interpreted in relation to the reported seed variability. Three repeats provide an initial estimate of stability, not a definitive statistical comparison. Generation metrics and the selected checkpoint should be re-evaluated on a larger sample before drawing conclusions about memorisation, diversity, or clinical suitability.

### 5. Future actionable experiment recommendations beyond the scope of this report

The current configuration (`codebook_size=512`, `ssim_weight=0.1`, and two downsampling stages) provides a useful reference for controlled follow-up experiments. The sweeps suggest the following changes for further investigation:

1. **Increase the codebook size to 1024.** This setting improved mean test SSIM, MSE, FD, precision, and sharpness ratio in the codebook sweep. The trade-off is reduced recall relative to 128 and no clear improvement in best-match SSIM. Evaluate whether the larger codebook improves anatomical detail without increasing training-image similarity or reducing sample diversity.

2. **Reduce the SSIM weight to 0.05.** Relative to 0.1, this setting slightly improved MSE and reduced the mean close-match rate from 0.0977 to 0.0898, with a small decrease in test SSIM. It is a promising candidate for reducing excessive similarity, but the close-match result has substantial seed variability and does not demonstrate privacy.

3. **Retain two downsampling stages.** Three stages increased parameter count from approximately 2.0M to 7.9M and degraded reconstruction and several generation metrics. Further architecture changes should focus on preserving spatial detail and avoiding artefacts rather than simply increasing depth.

4. **Evaluate the combined configuration.** The next experiment should combine `codebook_size=1024`, `ssim_weight=0.05`, and `downsample_stages=2`. This configuration combines promising results from the individual sweeps, but the joint effect is unknown because the hyperparameters were varied separately. Train it with the same data split, epoch budget, and seeds as the reference model. Compare the combined model against the current configuration and the best individual alternatives before accepting the change.

5. **Strengthen memorisation and privacy evaluation.** Run the full nearest-neighbour audit using `evaluate.py` on uncurated generated samples. Examine the closest training matches individually, compare similarity distributions rather than only a single threshold, and investigate whether particular patients or anatomical regions are disproportionately represented. Where feasible, add patient-level privacy tests and confirm that improvements persist across seeds.

6. **Prioritise anatomical plausibility and downstream safety.** Inspect uncurated sample grids, latent-space interpolations, and automatically flagged failures. Review tissue boundaries, spatial continuity, implausible structures, and artefacts with appropriate domain expertise. Where segmentation labels are available, assess anatomical consistency against them. Before using synthetic images for downstream training, compare downstream performance using real-only, synthetic-only, and mixed training data, and evaluate performance on held-out real patients, including relevant subgroup and pathology-specific analyses where the dataset supports them.

7. **Validate reproducibility and distributional metrics.** Increase the number of repeats and generated samples, report uncertainty, and retain the best checkpoint selected using validation data. Keep the test set reserved for final evaluation. Reassess FD, precision, recall, coverage, sharpness, close-match rate, and prior validation CE together rather than optimising a single metric.

These experiments would test whether the modest improvements observed in individual sweeps transfer to a combined model and whether they improve the properties that matter for safe synthetic medical imaging. A configuration should only be recommended for downstream training after demonstrating adequate anatomical plausibility, useful sample diversity, acceptable similarity to training data, and performance on held-out real data. The present results motivate these investigations but do not establish that any configuration is clinically safe or privacy-preserving.


## 8. Discussion and Recommendation

The VQ-VAE is the stronger model on this dataset: it improves reconstruction SSIM from `0.713` to `0.950`, reduces Fréchet distance from approximately `226` to `75.14`, and uses almost the entire codebook. Its feature distribution is closer to real HipMRI than the VAE's, but it remains different from the held-out real reference and still produces occasional artefacts.

The gain has a substantial cost: the VQ-VAE has nearly twice as many trainable parameters, takes about nine times longer to train, and has about 263 times the generation latency. This may be acceptable for small research cohorts, but is inefficient for high-throughput generation without prior optimisation.

The findings have important limits:

- **Generalisation:** only 38 patients were available, with three in the held-out test partition. Independent patients and imaging centres are needed for stronger generalisation evidence.
- **Anatomical validity:** no expert clinical review was performed. MSE, SSIM, Inception features, frequency measures, and t-SNE cannot establish physiological correctness.
- **Privacy:** no differential-privacy mechanism, membership-inference test, or patient re-identification evaluation was performed. Nearest-neighbour differences do not prove anonymity or safe sharing.
- **Volumetric consistency:** the model generates independent 2D slices and does not enforce continuity between adjacent slices.
- **Downstream utility:** the project did not test whether synthetic images improve a model trained for a medical-imaging task. A future study should compare downstream performance on held-out real patients with and without synthetic augmentation.

**Recommendation:** retain the VQ-VAE as the preferred experimental architecture for further research, but treat its outputs as research prototypes only. Do not regard them as clinically validated or privacy-guaranteed data. Before downstream use or sharing, obtain expert anatomical review, test patient-level generalisation and privacy risk, and evaluate utility on a separate task using held-out real patients. Optimising the autoregressive prior should be a priority if generation throughput matters.

## 9. Artificial Intelligence Usage Disclosure

### Tools used

Claude (Anthropic) was used throughout this project to assist in the generation of code for the models, figures and results mentioned throughtout this report based on human decisions as described by the changelog. ChatGPT was used in explaing concepts and results where no code needed to be generated.

### Tasks supported by AI

1. Initial scaffolding and implementation of `modules.py` and `dataset.py`, and the `train.py` training loop base on desisions made in the initial feasability specification.
2. Iterations on the initial model designs including converting the baseline model from an AE to VAE.
3. The production of tools such as `explore.py`, `evaluate.py` and its helps to assist in diagnosing problems and benchmarking solutions such that future decisions and analysis can be confidently made.
4. Constructing `predict.py` as well as drafting much of the initial feasability report
5. Drafting this README report based on visual and quantitaive data produced by the model.
6. Explaning core concepts throught the entire process to provide an overview of the specific concept, such as Autoencoders, codebook collapse ot SSIM, for consolation rather than reading in depth scientific papers.

### Review and verification

All code was manually reviewed ensuring that there were no obvious bugs or inaccuracies while ensuring strict version control so, that if any issues were later found to be introduced by the foreign code, changes can be rolled back to the last stable version. The models were then verified based on visual sample reconstruction and generations to ensure that the outputed samples match what is suggested by the numerical results. Finally, solutions to issues proposed to the AI model where tested to determing wheither the casue of the issue was identified or whether the changes made did not address the original issue. The original figures were refenced to ensure that all results matched the model output rather than being haloosinated by the LLM.

## 10. References

1. The University of Queensland. (2026). *COMP3710/COMP7710 Pattern Recognition: Version 2.5 Final Assessment Specification*.
2. van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural discrete representation learning. *Advances in Neural Information Processing Systems*, 30.
3. Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. *IEEE Transactions on Image Processing*.
4. PyTorch Team. (2026). *PyTorch Documentation* (Version 2.14). https://docs.pytorch.org/docs/2.14/index.html
