# Edge-Preserving CT Denoising: From First Principles

**Course:** PHC-351  
**Instructor:** Dr. Mayank Goswami  
**Prepared By:** Vinay  
**Enrolment No:** 24324026  
**Branch:** BS-MS Physics  

---

## 📖 Abstract
This project challenges the standard machine learning paradigm for image denoising. Rather than evaluating networks using Peak Signal-to-Noise Ratio (PSNR) on datasets infused with Additive White Gaussian Noise (AWGN), this research introduces a **physics-grounded, sub-pixel metrology approach**. We simulate actual scanner hardware (Poisson quantum mottle + Filtered Back-Projection) and build an exact analytical ground truth geometry. This enables the precise measurement of spatial boundary shifts, edge spread function (ESF) blur widths, and small-feature shrinkage bias caused by classical and neural denoising architectures.

---

## 🏗️ Project Architecture

The codebase is highly modular, explicitly separating physical simulation, mathematical metrology, and neural network topologies to ensure absolute data hygiene.

```text
project/
├── aiml-notebook.ipynb    # Main Jupyter/Kaggle notebook for execution and plotting
├── baselines.py           # Classical non-learning filters (Bilateral, Perona-Malik)
├── data.py                # Disjoint dataset generation, strict geometric augmentation, and caching
├── experiments.py         # Sub-pixel validation, empirical testing, and power-law fitting
├── loader.py              # Clinical LIDC-IDRI DICOM parsing and HU windowing
├── losses.py              # L1, MSE, GDL, SSIM, and Signed Distance Field (SDF) weighted calculus
├── metrics.py             # Normal Vector PCA and Half-Level Crossing metrology
├── models.py              # Parameter-matched U-Net, PlainCNN, and Hourglass architectures
├── phantom.py             # Generative analytical geometry and sub-pixel edges
├── scanner.py             # Beer-Lambert scaling, Poisson noise, and FBP reconstruction
├── test_phantom.py        # Unit tests ensuring exact geometric fidelity
└── train.py               # AdamW + AMP (FP16) training loop with decoupled FP32 loss calc
```

---

## 🔬 Methodology

### 1. Forward Physics & Scanner Simulation (`scanner.py`)
Training on Additive White Gaussian Noise is physically invalid for CT imaging. The inverse Radon transform (Filtered Back-Projection) utilizes a high-pass ramp filter ($|\omega|$) that catastrophically amplifies high-frequency detector noise. This forces CT noise to be streaky, anisotropic, and spatially correlated. Our forward pipeline accurately replicates this by simulating Poisson photon statistics before FBP reconstruction.

### 2. Sub-Pixel Metrology (`metrics.py`, `phantom.py`)
To avoid threshold-dependent heuristic edge detectors (like Canny/Sobel), we generate analytic phantoms from exact parametric equations. We measure spatial degradation using:
*   **Normal Vector PCA:** Identifies the exact orthogonal normal for every true boundary point to avoid pixel-grid aliasing.
*   **Half-Level Crossing:** Samples the 1D normal profile at 0.25-pixel increments to calculate the dynamic midpoint between material plateaus, yielding sub-pixel edge error and 10–90% ESF blur width.

### 3. Ablations & Topologies (`models.py`, `losses.py`)
To isolate structural effects, three residual networks were mathematically parameter-matched (~1.93M parameters):
1.  **Plain CNN:** 8-layer flat convolution (No downsampling, no skips).
2.  **Hourglass:** Encoder-decoder (No skips).
3.  **U-Net:** Multiscale hierarchy with symmetric feature concatenation skips.

The loss functions combine **L1**, **Gradient Difference Loss (GDL)** to preserve derivative steepness, and **Signed Distance Field (SDF)** weights to aggressively penalize sub-pixel boundary errors.

---

## 🚀 Kaggle Execution Guide

To reproduce the findings, use a Kaggle Notebook configured with a **T4 GPU**.

### Step 1: Mount the Data and Code
1. Click **+ Add Input** in the right-hand panel of your Kaggle session.
2. Mount the clinical dataset by searching for: `LIDC_project11_dataset`
3. Mount the Python scripts by uploading the project folder as a dataset (e.g., `project11_2026`).

### Step 2: Environment Initialization
Kaggle Notebooks cannot import custom `.py` files directly from read-only input directories. Run the following code block in your first cell to copy the files to the working directory:

```python
import glob, shutil, sys, os, subprocess, torch

# Copy all source files to the writable /kaggle/working/ directory
for f in glob.glob("/kaggle/input/**/*.py", recursive=True): 
    shutil.copy(f, "/kaggle/working/")

# Insert path for module resolution
sys.path.insert(0, "/kaggle/working")
os.chdir("/kaggle/working")

# Install missing clinical dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pydicom"])
print(len(glob.glob("*.py")), "py files | GPU:", torch.cuda.is_available())
```

### Step 3: Run the Pipeline
Execute the modules sequentially as structured in `aiml-notebook.ipynb`:
1.  **Unit Tests:** Verify the physics and metrology engines (`%run scanner.py`, `%run metrics.py`).
2.  **Dataset Generation:** Build the strict disjoint splits using `data.build_split()`. (Takes ~5-12 minutes; caches to disk).
3.  **Baseline Tuning:** Optimize the classical filters independently on validation data using `experiments.tune_filter()`.
4.  **Model Training:** Train all 13 ablation configurations. AMP optimization runs the U-Net at ~1.5s/epoch.
5.  **Evaluation & Findings:** Execute the evaluation suite on unseen test seeds and OOD noise distributions. Outputs are saved to `results.csv`.

---

## 📊 Summary of Key Findings

*   **Finding A (PSNR is Flawed for Metrology):** The Plain CNN achieved the highest PSNR (44.65 dB) but failed drastically at structural preservation. The classical edge-tuned Bilateral filter achieved the lowest mean edge error (0.077 px) but smeared edges completely (3.709 px blur width). *No single metric picks the right model.*
*   **Finding B (Out-of-Distribution Robustness):** Under severe photon starvation ($N_0 = 1000$), the deep Hourglass bottleneck completely collapsed (0.340 px error, identical to the noisy input), proving that deep networks without skip connections fail spatial localization.
*   **Finding C (Small-Feature Bias):** The PSNR-optimal Plain CNN systematically eroded high-curvature boundaries, shrinking small circular voids by -0.18 pixels (double the raw noise error). The U-Net corrected this to -0.08 pixels.
*   **Clinical Transfer:** Zero-shot application on real LIDC-IDRI chest CT scans yielded a +2.9 dB structural PSNR improvement, validating the network's ability to leverage piecewise-flat anatomical priors from the synthetic phantoms.