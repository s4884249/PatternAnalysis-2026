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

## Dependencies

- numpy 2.14.1
- torch
- nibabel
- scikit-image
- matplotlib

## References
