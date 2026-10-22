# Image Generation: Synthetic Medical Cohorts and Privacy using VQ-VAE

## Project Description

### User Need

### Scope

### Acceptance criteria

### Dataset

## File Structure

```
recognition/VQ-VAE_s4884249/
├── dataset.py
├── images
│   └── ...
├── modules.py
├── params.py
├── predict.py
├── README.md
├── requirements.txt
└── train.py
```

| File              | Purpose |
| ----------------- | ------- |
| modules.py        | Basic convolutional autoencoder, VQ-VAE, vector quantizer and masked-convolution autoregressive code prior |
| dataset.py        | NIfTI data audit, MRI/mask classification review, intensity preprocessing and group-level data splitting |
| train.py          | Training, validation, checkpoint selection, test metrics, codebook statistics and resource tracking |
| predict.py        |Sample generation, reconstruction grids, latent interpolation, nearest-neighbour audits and failure-case visualisations |
| params.py         | Model parametes |
| utils.py          | Helper utilities |
| README.md         | Comprehensive report and experiment documentation |
| requirements.txt  | Python required libraries |

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Model Architecture and Difficulty

### VQ-VAE Advantages

### VQ-VAE Disadvantages

## Data Augmentation

## Preliminary Feasibility Evidence

## Risks, Budget & Fallback

## Changelog

Some changes were made to the Preliminary models when issues were identified which would prevent
the task to be completed successfully with the project's scope.

### Version 1

The AE generated samples by encoding training images, finding nearby latent representations,
interpolating between pairs of those representations, and decoding the interpolations. This
preserved the spatial structure of real latent maps much better than independent Gaussian
noise and the sampler was reproducible with a fixed seed.

![Generated samples from nearby latent representations](./images/generated_samples_from_nearby_latent_representations.png)
_Basic Auto-Encoder: Generated samples from nearby latent representations_

However, a basic deterministic AE is not a probabilistic generative model. Latent interpolation
should work, but the outputs may resemble training images or contain unrealistic blends. The new
sample grid and nearest-neighbour results should be inspected before treating them as valid
synthetic MRI data. To properly compare the model with the VQ-VAE, it was upgraded to a VAE such
that the latent space was regularised so sampling can produce more meaningful outputs (and new).

> Takeaway
> To achieve a reasonable baseline, a VAE was used rather than a basic AE such that new samples
> can be more reliably generated from the latent distribution as required to fulfill ethical
> and task constraints (i.e. generating truly new samples rather than recreating existing images).

## Dependencies

- numpy 2.14.1
- torch
- nibabel
- scikit-image
- matplotlib

## References
