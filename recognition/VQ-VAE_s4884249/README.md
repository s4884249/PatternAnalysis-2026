# Image Generation: Synthetic Medical Cohorts and Privacy using VQ-VAE

## Project Description

Medical imaging research is constrained by stringent patient data privacy laws (HIPAA, GDPR) and limited clinical scan availability. Generative models allow synthesizing realistic synthetic cohorts for data augmentation and anonymized multi-centre sharing. However, healthcare synthesis carries unique hazards: the model must preserve authentic physiological tissue boundaries without mode collapse, and must not hallucinate anatomically impossible structures or memorize training scans (which would violate patient confidentiality).

### User Need

The intended user is a medical imaging researcher or machine learning engineer evaluating whether synthetic images can supplement real HipMRI data. The prototype will implement a Vector-Quantised Variational Autoencoder (VQ-VAE) to generate synthetic 2D HipMRI slices and investigate the trade-off between image fidelity, sample diversity, and memorization of training data.

The key dilemma is determining whether scans that are anatomical valid without becoming blurry or facing mode collapse, hallucinating physiologically impossible tissue junctions or non-existent pathologies and without producing verbatim memorization of training patient scans.

If a generative model simply memorizes training scans, sharing the synthetic dataset constitutes an illegal breach of patient confidentiality. However, if it hallucinates unrealistic tissue structures, downstream models trained on it will learn hazardous diagnostic artifacts.

### Scope

The project is scoped to 2D HipMRI image generation using the provided `keras_slices_data` dataset. It will compare VQ-VAE against a simpler Autoencoder baseline under consistent data splits and evaluation conditions.

### Acceptance criteria

- [x] Train a VQ-VAE that can reconstruct input HipMRI slices.
- [ ] Train a VQ-VAE that can generate new samples through its learned discrete latent representation.
- [x] Evaluate generated or reconstructed images using Structural Similarity Index (SSIM), targeting a value above `0.6`
- [ ] Compare VQ-VAE against a basic autoencoder using the same data split and comparable evaluation procedures, reporting quantitative results and representative image grids.
- [ ] Audit generated samples against training images using nearest-neighbour comparisons and inspect latent-space interpolations to investigate memorization, mode collapse, and anatomical plausibility.
- [ ] Record training duration, peak GPU memory usage, model parameter count, and generation runtime. Establish a practical training budget and retain a checkpoint from the best validation run.

## Preliminary Feasibility Evidence

### Initial Data Audit

- [x] Confirm that `keras_slices_data` is accessible in the COMP3710 environment and determine the number of available images.
- [x] Inspect image dimensions, intensity ranges, data types, missing or corrupted images, and representative samples.
- [x] Check whether patient or source-volume identifiers are available to support leakage-resistant data splitting.
- [x] Display a sample grid to assess anatomical content, image quality, and the preprocessing required by the model.

#### Preliminary Results

The initial audit was conducted on Rangpur using the provided `keras_slices_data` directory. The directory was accessible, and `25,320` candidate `NIfTI` files were inspected successfully, with no unreadable files or detected non-finite voxel values. All files used `float32` data. The observed dimensions were `256 × 128` for `25,200` files and `256 × 144` for `120` files. The global observed intensity range was `[0, 1212]`.

![Sample Grid](images/preliminary_sample_grid.png)

The sample grid indicated that the directory contains both MRI images and segmentation masks. Therefore, the total file count cannot yet be interpreted as the number of usable MRI training images. No separate metadata-like files were identified, so patient or source-volume identifiers must be validated from the filenames or dataset documentation before creating leakage-resistant splits.

The initial results support proceeding with data characterization and a small training smoke test. Before full training, the MRI and segmentation files will be separated, the differing image dimensions investigated, and a patient-level train/validation/test split established. Intensity normalization and model input dimensions will be selected based on the resulting MRI-only statistics.

### Baseline Smoke Test

A small basic autoencoder will first be trained for a few epochs to verify data loading, GPU execution, loss computation, and image reconstruction. This will establish a working pipeline before introducing vector quantization.

![Reconstruction Grid](images/preliminary_reconstruction_grid.png)

- [x] The training and validation losses remain finite.
- [x] The model can reconstruct input images without tensor-shape or numerical errors.
- [x] GPU memory consumption and iteration time are manageable.
- [x] Model checkpoints can be saved and loaded successfully.

### Preliminary Resource Estimate

The project will use a MacBook Pro (M5 Pro, 64GB) rather than relying on the limited memory of a local computer. The initial resource budget is a provisional planning estimate, not a measured benchmark.

Activity | Proposed budget
--- | ---
Data audit and preprocessing | 1–2 hours
Baseline and VQ-VAE smoke tests | 1–2 GPU hours
Initial hyperparameter experiments | 2–4 GPU hours
Final training and evaluation | 2–4 GPU hours
--- | ---
Initial total budget | 6–12 GPU hours

## Risks, Budget & Fallback

### Key Technical Risks

| Risk | Mitigation
| --- | ---
| Poor reconstruction quality | Inspect reconstruction grids, verify normalization and loss scaling, and tune model capacity and learning rate.
| Codebook collapse or limited codebook usage | Monitor the number of active codebook entries and | quantization behaviour; adjust codebook size and optimization settings if necessary.
| Memorization of training images | Keep validation and test data separate, compare generated images with nearest training neighbours, and inspect unusually close matches.
| Mode collapse or limited sample diversity | Generate uncurated sample grids across multiple random seeds and assess whether outputs show meaningful variation.
| Data leakage between slices | Split by patient or source volume where identifiers are available, rather than randomly splitting individual slices from the same subject.
| Excessive training time or GPU memory use | Begin with low-resolution inputs and a compact model, measure peak GPU memory, and increase model capacity only when justified by results.
| Unreliable evaluation | Compare the VQ-VAE and baseline using consistent data splits and clearly distinguish reconstruction metrics from metrics on newly generated samples.

### Computing Budget and Experiment Plan

The initial allocation is 6–12 GPU hours on Rangpur, with additional time requested only if the measured training rate or evaluation results justify it. Experiments will be recorded with their configuration, random seed, training duration, validation loss, SSIM, peak GPU memory, and checkpoint location.

The first experiment will be the data audit and baseline smoke test. If that succeeds, a compact VQ-VAE will be trained and evaluated before conducting a small hyperparameter search. The final configuration will be selected using validation results, and the held-out test set will be used for final reporting.

### Fallback Strategy

If the initial VQ-VAE is unstable or exceeds the computing budget, the first fallback will be to reduce input resolution, channel widths, latent dimensionality, or the number of training epochs. The codebook and quantization mechanism will be retained so that the implementation still addresses the selected Hard Difficulty task.

If quantization remains unstable, the basic autoencoder will serve as a functioning baseline for diagnosing the problem and comparing reconstruction quality. The implementation will document the limitation and any unsuccessful experiments rather than claiming that the VQ-VAE met the acceptance criteria without evidence.

If synthetic image quality remains poor, the project will focus on diagnosing the failure through reconstruction comparisons, codebook usage, sample diversity, and nearest-neighbour analysis. The final engineering recommendation
will state whether the evidence supports using the generated images and identify the limitations that remain.

The project is feasible in principle because it uses 2D images and a convolutional encoder-decoder rather than a computationally heavier hierarchical generator. However, feasibility will be confirmed only after the dataset audit and measured smoke test. The key success criterion is not merely producing images, but demonstrating their fidelity, diversity, and lack of obvious training-data memorization through reproducible experiments.

Actual training duration will depend on dataset size, image resolution, model configuration, and GPU utilization. The budget will be revised after measuring the smoke-test throughput. Full training will proceed only after the preliminary results indicate that the target is achievable within the available resources.

## Dataset

The project will use the provided HipMRI 2D slices from keras_slices_data, available through the COMP3710 project environment. Images will be loaded, checked for corrupted or inconsistent dimensions, and normalised to a consistent intensity range suitable for the selected reconstruction loss.
![Sample Slices](images/explore_sample_slices.png)_Sample slices for each train/validate/test set_

The scans are pre-split into three distinct directories, which will be respected, for training validation and testing with no obvious difference in the makeup scans allocated for each which could negativly impact the models validation and test scores. A split such as if all the training scans were made up of a single person, the model would lean to produce the features of that person rather than of the dataset as a whole.
![Pixel Statistics](images/explore_pixel_statistics.png)_Pixel Statistics acros all samples_

The sample images follow a general pattern, as shown by the mean slice vaugely resembling a scan with particular learnable features differing the various which the model will need to attemp to recreate without simply reproducing the proteted training data.

```python
# See `dataset.py`
def normalize_mri(array: np.ndarray) -> np.ndarray:
    """Per-slice scaling to [0, 1] after clipping to the 1st-99.5th percentile.

    Clipping stops a few very bright voxels from compressing the useful range, and
    per-slice scaling removes the arbitrary absolute intensity scale between scans.
    """
    low, high = np.percentile(array, [1.0, 99.5])
    if high <= low:  # constant slice: nothing to scale
        return np.zeros_like(array, dtype=np.float32)
    return ((np.clip(array, low, high) - low) / (high - low)).astype(np.float32)
```

The function `normalize_mri` is used to ensure the samples are normalised to a consistent intensity and avoid a few very bright voxels from compressing the useful range.

```console
== Files ==
manifest rows: 25320 | readable: 25320 | unreadable: 0
images: 12660 | masks: 12660

== Images and masks per split ==
train images  11460 | masks  11460
val   images    660 | masks    660
test  images    540 | masks    540

== Cases (patients) ==
train   31 cases | slices per case: min 60 | median 480 | mean 369.7 | max 480 | std 171.8
val      4 cases | slices per case: min 60 | median 60 | mean 165 | max 480 | std 181.9
test     3 cases | slices per case: min 60 | median 60 | mean 180 | max 420 | std 169.7
cases shared between splits: none (no leakage)

== Time points (weeks) ==
slices per week: {0: 2280, 1: 1620, 2: 1560, 3: 1560, 4: 1500, 5: 1440, 6: 1320, 7: 1380}
weeks per case: min 1 | median 8 | mean 5.553 | max 8 | std 3.151

== Image / mask pairing ==
images with a mask: 12660 / 12660 | masks without an image: 0

== Raw geometry ==
shape 256x128: 12600 (99.5%)
shape 256x144: 60 (0.5%)

== Raw intensities (per slice, before normalisation) ==
global min 0 | global max 1212
slice max         : min 196 | median 396 | mean 414.1 | max 1212 | std 110.7
slice mean        : min 26.87 | median 69.17 | mean 72.59 | max 268.2 | std 24.71
slice std         : min 35.68 | median 68.97 | mean 73.37 | max 226.6 | std 22.51
slice nonzero_frac: min 0.6934 | median 0.9087 | mean 0.8974 | max 0.957 | std 0.04143
(normalisation clips each slice to its 1st-99.5th percentile and rescales to [0, 1], so the raw scale varies freely between slices)

== Preprocessed pixels (500 random training slices, 128x80) ==
pixel mean 0.1719 | pixel std 0.2138
background (pixels < 0.05): 39.7%
per-slice mean: min 0.1012 | median 0.1622 | mean 0.1719 | max 0.2863 | std 0.04155
```

Overall there are `12660` usable images broken into an approximate 90/5/5 percent train/validate/test split that will be utilised throughout the project. Where patient identifiers or source-volume information are available, slices from the same patient or volume will be kept within a single split to reduce data leakage. The training set will be used to learn model parameters, the validation set to select configurations and checkpoints, and the test set reserved for final evaluation.

```python
# See `dataset.py`
def load_manifest(manifest_path: Path, data_root: Path, keep_shape: str = SOURCE_SHAPE_TEXT) -> List[Dict[str, str]]:
    """Return the readable image rows (path, split, group_id); masks, error rows and
    images whose shape is not `keep_shape` (default 256x128) are dropped."""
    data_root = data_root.expanduser().resolve()
    rows = []
    dropped_shapes: Counter = Counter()
    for source in read_manifest(manifest_path):
        if source["kind"] != "image" or source["error"]:
            continue
        if keep_shape and source["shape"] != keep_shape:
            dropped_shapes[source["shape"] or "unknown"] += 1
            continue
        if not (data_root / source["path"]).is_file():
            raise FileNotFoundError(f"Manifest file missing under --data-root: {data_root / source['path']}")
        rows.append({k: source[k] for k in ("path", "split", "group_id")})
    if dropped_shapes:
        print(f"Dropped images whose shape is not {keep_shape}: {dict(dropped_shapes)}; kept {len(rows)}")
    if not rows:
        raise ValueError(f"The manifest contains no readable {keep_shape} image rows")
    return rows

```

```console
Dropped images whose shape is not 256x128: {'256x144': 60}; kept 12600
```

To avoid dealing with sample images of to different sizes and the streching or black bars produced in an attempt of normalisation, the `0.5%` of images in the datase with a `256x144` resolution were ommited from the data dataset resulting in a final count of `12600` usable images.

## File Structure

```console
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

!!TODO!! Fix above tree and purposed

## Installation Instructions

```shell
# Install Python 3.14 (use appropriate package manager)
brew install python@3.14

# Create virtual enviroment
/opt/homebrew/bin/python3.14 -m venv .venv

# Activate virtual enviroment
source .venv/bin/activate

# Install required packages
python -m pip install -r requirements.txt
```

## Model Architecture and Difficulty

Model: `Vector-Quantised Variational Autoencoder (VQ-VAE)`
Difficulty band: `Hard Difficulty`

VQ-VAE uses an encoder to map an image into a lower-dimensional latent representation. The representation is quantized by assigning latent vectors to entries in a learned discrete codebook. A decoder then reconstructs the image from the quantized representation. The training objective combines reconstruction loss with terms that train the codebook and encourage the encoder to use its discrete representations effectively.

The discrete latent representation provides a way to learn recurring image structures, but it can also introduce reconstruction artifacts or underuse of the codebook. These behaviours will be investigated through reconstruction comparisons, codebook usage statistics, and generated sample grids.

The implementation will use PyTorch and run on a MacBook Pro (M5 Pro, 64GB). The architecture will begin with a relatively small convolutional encoder and decoder, with the latent dimensionality, codebook size, and reconstruction objective selected to suit the resolution and characteristics of the dataset.

### VQ-VAE Advantages

!!TODO!!

### VQ-VAE Disadvantages

!!TODO!!

### Relevant Course Concepts and Literature

The project applies convolutional neural networks, representation learning, encoder-decoder architectures, reconstruction losses, latent representations, optimization, validation, and quantitative model evaluation.

The assessment specification identifies VQ-VAE as a Hard Difficulty model for HipMRI 2D slices and cites "Neural discrete representation learning" by NeurIPS (2017). A basic autoencoder will provide the required simpler generative baseline. The comparison will examine whether discrete latent quantization provides useful improvements in image structure and diversity relative to the simpler model.

The central research question is whether VQ-VAE can generate anatomically plausible and sufficiently diverse synthetic images without reproducing training examples. The final recommendation will be based on experimental evidence rather than visual quality alone.

## Data Augmentation

!!TODO!!

```python
# !!TODO!!
```

## Changelog

Some changes were made to the Preliminary models when issues were identified which would prevent the task to be completed successfully with the project's scope.

### Version 1

The AE generated samples by encoding training images, finding nearby latent representations, interpolating between pairs of those representations, and decoding the interpolations. This preserved the spatial structure of real latent maps much better than independent Gaussian noise and the sampler was reproducible with a fixed seed.

![Generated samples from nearby latent representations](images/version1_vae_generated_samples.png)
_Basic Autoencoder: Generated samples from nearby latent representations_

However, a basic deterministic AE is not a probabilistic generative model. Latent interpolation should work, but the outputs may resemble training images or contain unrealistic blends. The new sample grid and nearest-neighbour results should be inspected before treating them as valid synthetic MRI data. To properly compare the model with the VQ-VAE, it was upgraded to a VAE such that the latent space was regularised so sampling can produce more meaningful outputs (and new).

> **Takeaway**
To achieve a reasonable baseline, a VAE was used rather than a basic AE such that new samples can be more reliably generated from the latent distribution as required to fulfill ethical and task constraints (i.e. generating truly new samples rather than recreating existing images).

### Version 2

Rather than having a deterministic AE, a VAE is now utilised to provide (in theory) genuinly new image samples.

![Generated samples from Gaussian](images/version2_vae_generated_samples.png)
_Basic Autoencoder: Generated samples from nearby latent representations_

However, after 50 epochs the generated samples still apeared to resemble random noise.

> **Takeaway**
In order to make improvements, occationaliy it may be required to step back. After investigation the issue with the new model will be found and the final result should be more useful than the AE's samples but the progress looks a first like a step backwards.

### Version 3

To optimise the results and identify causes for model collapse, detailed hyper-parameter adjustments must be made. To achive this, the code was modified to provide up to 10 sample reconstructions over the course of the training process such that the point at which the training should stop could be determined as not to either over-fit/under-fit the training data.

![Reconstruction by Epoch](images/version3_vae_reconstructions_by_epoch.png)
_VAE - Reconstruction by Epoch_

![Reconstruction by Epoch](images/version3_vq-vae_reconstructions_by_epoch.png)
_VQ-VAE - Reconstruction by Epoch_

However, the model did not support the use of Apple Silicon GPU (MPS) caused the training to take an excessive amount of time, leading to difficulty in efficently comparing hyper-parameters. As such, before further progress could be made MPS support needed to be added.

> **Takeaway**
Regardless of the effeciancey of the model, the performance depends on the availability of appropriate hardware. The model could use the latest inovation in mathematics and computer science but without appropriate hardware it will be slow and possibly unusable if the use case is time sensitive.

### Version 4

Apple Silicon GPU (MPS) support was added such that the models could be efficently run on the MacBook Pro being used for training. 

![CPU](images/version3_speed.png)
_CPU Version_
![GPU](images/version4_speed.png)_GPU Version_

Now, experements can be run with a larger epoch count for more reliable results and to determine if the computational power's constraint on traning time was the limiting factor.

![Generated Samples](images/version4_vae_generated_samples.png)
_VAE - Generated Samples_

![Generated Samples](images/version4_vq-vae_generated_samples.png)
_VQ-VAE - Generated Samples_

As obseved above, thile the VQ-VAE was able to generate passable samples, the original issue of the VAE producing a poor result has not beens solved. Now that the model can be trained efficently, the core issue can be identified more quickly.

> **Takeaway**
Increasing or decreasing computing power will not affect the quality or accuracy of the generated sample.

### Version 5

In addition to now dropping the `256x144` images and maintaining the original aspect ration, the VAE issue with the generation of new samples was solved. The main problem was that the old VAE had too much spatially independent latent information and no mechanism to learn the spatial correlations needed for realistic anatomy. It reconstructed images well but was not necessarily able to generate realistic new ones.

The VAE was changed from a high-dimensional spatial latent representation to a compact, globally regularised latent vector, with revised KL scaling to improve the potential quality of random samples.

The latent representation eas changed from a spatial latent map of `64 × 16 × 10 = 10,240` values to a single global latent vector of `256` values per image. This creates a stronger bottleneck and encourages the model to represent the overall image structure. The KL loss was adjusted to encourage latent vectors to follow a standard normal distribution, N(0,I), making random sampling more consistent with the decoder's training di stribution. Finally, the KL term was changed to be summed over latent dimensions and divided by the number of image pixels. The default kl_beta is now `0.01`, providing a tunable balance between reconstruction quality and latent regularisation.
Results and expected impact

![Generated Samples](images/version5_vae_generated_samples.png)
_VAE - Generated Samples_

![Generated Samples](images/version5_vq-vae_generated_samples.png)
_VQ-VAE - Generated Samples_

Overall, the generated anatomy was more coherent because the decoder receives a compact global representation rather than an independently sampled spatial latent map. The trade-off being that the MSE reconstruction loss may still produce blurry images. Additionally, with the `256x144` samples removes, there are no longer black bars on the edges.

## Training

### Usage Instructions

```shell
# !!TODO!!
```

### !!TODO!! Test Name

## Training Results

!!TODO!!

## Generated Images

!!TODO!!

## Training Loss

!!TODO!!

## Benchmarking

### !!TODO!! Some kind of analysis

## Dependencies

- python 3.14.7
- numpy 2.5.4
- torch 2.14.1
- nibabel 5.4.2
- scikit-image 0.26.0
- matplotlib 3.11.2

## AI Disclosure

**!!TODO!!**
Mandatory AI Disclosure Section: In your README.md, you must include a dedicated subsection
titled “Artificial Intelligence Usage Disclosure” documenting:
15
• Which AI tools were utilized (e.g., GitHub Copilot for auto-complete, ChatGPT/Claude for
debugging syntax, Gemini for concept lookup).
• What specific tasks AI assisted with (e.g., scaffolding data loading functions, refactoring plotting
scripts).
• How you audited, verified, and empirically tested the AI-suggested code.
## References
