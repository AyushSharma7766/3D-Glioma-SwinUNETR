# High-Performance 3D Swin-UNETR for Whole-Brain MRI Segmentation
### Optimized for Local NVIDIA H200 GPU (Hopper Architecture, 141 GB HBM3e) with MONAI, PyTorch AMP, and MLflow

---

## 1. Executive Summary & Architectural Overview

This repository provides a modular, production-ready, and academically defended implementation of the **3D Swin-UNETR** (Swin Transformer-based U-Net) architecture tailored for whole-brain structural MRI segmentation. 

Whole-brain anatomical segmentation requires classifying both macro-scale tissue compartments (cerebral white matter, cortical gray matter) and micro-scale subcortical structures (thalamus, caudate, putamen, hippocampus, amygdala, lateral and 3rd ventricles). The pipeline is explicitly optimized to saturate the compute capabilities of the **NVIDIA H200 Tensor Core GPU (141 GB HBM3e, 4.8 TB/s memory bandwidth)** using **Automatic Mixed Precision (AMP `bfloat16`)**, high-throughput **MONAI dictionary transforms**, **sliding window Gaussian inference**, and **MLflow** experiment tracking.

```
                           +----------------------------------------+
                           |   Full 3D NIfTI Brain MRI Volume       |
                           |   (e.g., 256 x 256 x 176, T1w MPRAGE)  |
                           +----------------------------------------+
                                               |
                               [MONAI Dictionary Pipeline]
                               - EnsureChannelFirstd
                               - Orientationd ('RAS')
                               - Spacingd (1.0 mm isotropic)
                               - NormalizeIntensityd (Z-score)
                                               |
                       +-----------------------+-----------------------+
                       |                                               |
              [Training Pipeline]                            [Validation Pipeline]
          RandCropByPosNegLabeld (96^3)                    Full Native Resolution Volume
          1:1 Positive/Negative Patches                                |
          Spatial & Intensity Augmentations                            |
                       |                                               |
                       v                                               v
        +-----------------------------+              +-----------------------------------+
        |  3D Swin-UNETR Architecture |              |    Sliding Window Inference       |
        |  - Hierarchical Shifted W-MSA| <----------- |    - Overlap: 50%                 |
        |  - 48-dim Base Embedding    |  (Eval Mode) |    - 3D Gaussian Weight Blending  |
        |  - 4 Pyramid Stages         |              |    - Seam-Free Stitching          |
        +-----------------------------+              +-----------------------------------+
                       |                                               |
         [Hardware Acceleration: H200]                                 v
         - torch.amp.autocast('bfloat16')                     +--------------------+
         - Hopper 4th-Gen Tensor Cores                        |  Multi-Class Dice  |
         - Max Batch Size (141 GB HBM3e)                      |  Metric Evaluation |
                       |                                      +--------------------+
                       v                                               |
         +----------------------------+                                v
         |      DiceCELoss            |                     +----------------------+
         | (Soft Dice + Cross Entropy)|                     |   MLflow Tracking    |
         +----------------------------+                     | - Train Loss         |
                       |                                    | - Val Mean Dice      |
                       v                                    | - Per-Class Metrics  |
            [AdamW + Cosine Warmup]                         | - Model Checkpoints  |
                                                            +----------------------+
```

---

## 2. Academic Defense: Theoretical & Architectural Justifications

### 2.1 The Duality Problem in 3D Neuroimaging: CNNs vs. Standard ViT vs. Swin-UNETR
* **The Failure of 3D CNNs (3D U-Net / V-Net):** Standard 3D convolutions employ localized kernels ($3 \times 3 \times 3$). The receptive field grows linearly with depth. However, brain morphology exhibits critical long-range bilateral symmetries (e.g., the position of the left lateral ventricle is intimately correlated with the contralateral right ventricle across hundreds of voxels). CNNs struggle to encode these distant spatial interdependencies without extreme downsampling, which destroys fine subcortical boundaries.
* **The Quadratic Catastrophe of Standard Vision Transformers (ViT / UNETr):** Standard self-attention projects all tokens globally:
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
  For a modest volumetric patch of $96 \times 96 \times 96$ with $2 \times 2 \times 2$ patch tokens, the number of tokens is $N = 48^3 = 110,592$. Computing an $N \times N$ attention matrix requires $\approx 1.22 \times 10^{10}$ floating-point operations per attention head per layer, causing immediate Out-of-Memory (OOM) failures.
* **The Swin Transformer Solution (Shifted Window Multi-Head Self-Attention):**
  Swin-UNETR partitions the 3D volume into local, non-overlapping 3D sub-volumes (windows) of size $M \times M \times M$ ($7 \times 7 \times 7$). Attention is computed strictly within each window:
  $$\text{Complexity} = \mathcal{O}\left(4 \cdot M^3 \cdot N + 2 \cdot M^6\right)$$
  which scales **strictly linearly** $\mathcal{O}(N)$ with the total number of volumetric tokens. To cross window boundaries, alternate layers execute **Cyclic Shifted Window Self-Attention (SW-MSA)** by translating coordinates by $\left(\lfloor \frac{M}{2} \rfloor, \lfloor \frac{M}{2} \rfloor, \lfloor \frac{M}{2} \rfloor\right)$, establishing global multi-scale connectivity with zero parameter overhead.

---

### 2.2 Data Engineering Justifications
* **`Spacingd(pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest"))`:** Clinical MRI acquisitions are frequently anisotropic (e.g., $0.4 \times 0.4 \times 3.0\text{ mm}$ thick slices). Anisotropic voxels skew derivative calculations in 3D convolutional kernels and distort isotropic attention windows. Bilinear interpolation preserves continuous intensity gradients for the MRI, while Nearest-Neighbor interpolation prevents artificial label creation along tissue boundaries.
* **`Orientationd(axcodes="RAS")`:** Standardizes coordinate headers across multi-vendor scanners (Siemens, GE, Philips) into canonical Right-Anterior-Superior alignment.
* **`RandCropByPosNegLabeld`:** Over 75% of a head MRI consists of non-brain air, skull, and neck tissue. Unconstrained uniform random sampling causes gradient starvation on small anatomical structures. Enforcing an exact 1:1 ratio between foreground anatomical structures and background patches guarantees that every mini-batch contains informative structural boundaries.

---

### 2.3 Loss Formulation: Theoretical Synergy of `DiceCELoss`
$$\mathcal{L}_{\text{Total}} = \lambda_{\text{Dice}} \mathcal{L}_{\text{Dice}} + \lambda_{\text{CE}} \mathcal{L}_{\text{CE}}$$

* **Soft Dice Loss ($\mathcal{L}_{\text{Dice}}$):**
  $$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_i p_i y_i + \epsilon}{\sum_i p_i^2 + \sum_i y_i^2 + \epsilon}$$
  Because Dice measures harmonic overlap, it is completely scale-invariant. Misclassifying 100 voxels in the hippocampus (a small structure) incurs an equally severe penalty as misclassifying thousands of voxels in cerebral white matter, directly counteracting class imbalance.
* **Cross-Entropy Loss ($\mathcal{L}_{\text{CE}}$):**
  $$\mathcal{L}_{\text{CE}} = - \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{ic} \log(p_{ic})$$
  Dice loss exhibits flat plateau regions with near-zero gradients when predictions have zero initial overlap with ground truth. Cross-Entropy provides steep, convex, continuous gradients far from the optimum, accelerating early convergence.

---

### 2.4 Inference: Sliding Window Inference with Gaussian Importance Blending
Evaluating whole-brain volumes at native resolution (e.g., $256 \times 256 \times 176$) avoids interpolation blur. Moving a patch of size $96^3$ across the volume with 50% overlap creates boundary stitching seams if patches are averaged naively. 

MONAI's `sliding_window_inference` with `mode="gaussian"` weights every voxel prediction according to a 3D Gaussian kernel:
$$w(x, y, z) = \exp\left( - \frac{(x - c_x)^2 + (y - c_y)^2 + (z - c_z)^2}{2 \sigma^2} \right)$$
Central patch voxels (which benefit from maximal receptive field context) carry peak weight ($w=1.0$), while edge voxels (which suffer from padding artifacts) are down-weighted, synthesizing seamless full-brain segmentation masks.

---

### 2.5 Hardware Optimization on NVIDIA H200 (Hopper Architecture)
* **141 GB HBM3e Memory:** Enables volumetric patch sizes of $96^3$ or $128^3$ and mini-batch sizes of $4 \text{ to } 8$, stabilizing batch statistics and removing the need for gradient checkpointing.
* **Hopper 4th-Gen Tensor Cores with Native `bfloat16`:**
  Unlike FP16 (which has only 5 exponent bits and frequently underflows, requiring dynamic `GradScaler`), BF16 maintains the exact 8-bit dynamic exponent range of FP32. Gradient underflow/overflow is eliminated, allowing unscaled backpropagation directly on H200 Tensor Cores with zero scalar synchronization overhead.
* **Host Caching (`CacheDataset`):** Caches deterministic preprocessing in system RAM, preventing CPU I/O starvation and keeping H200 compute saturation at $>95\%$.

---

## 3. Repository Structure

```
.
├── config.py                 # Centralized configuration dataclasses (Model, Data, Hardware, Training)
├── train.py                  # Main training entrypoint with CLI argument parsing & lifecycle orchestration
├── evaluate.py               # Full-volume sliding window evaluation & NIfTI segmentation mask exporter
├── generate_sample_data.py   # CLI tool to synthesize multi-class brain MRI NIfTI phantoms
├── requirements.txt          # Pinned production dependencies
├── README.md                 # Academic defense documentation & execution guide
├── data/
│   ├── transforms.py         # MONAI dictionary transforms (LoadImaged, Spacingd, RandCropByPosNegLabeld)
│   ├── dataset.py            # CacheDataset & high-throughput DataLoader construction
│   └── synthetic.py          # Realistic 3D multi-class brain phantom generator
├── models/
│   └── swin_unetr.py         # SwinUNETR instantiation, FLOPs analysis & torch.compile integration
├── training/
│   ├── loss.py               # DiceCELoss configuration with mathematical justifications
│   └── trainer.py            # H200 training engine with AMP bfloat16, MLflow tracking, & checkpointing
└── evaluation/
    └── inferer.py            # Sliding window Gaussian blending evaluator & multi-class Dice metric
```

---

## 4. Quickstart & Execution Guide

### 4.1 Environment Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify GPU and PyTorch Hopper support
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"
```

---

### 4.2 Step 1: Immediate Verification with Synthetic Brain Volumes (Dry Run)
If real clinical NIfTI files are not yet downloaded, the pipeline **automatically synthesizes** realistic 3D brain volumes with multi-class anatomical structures:

```bash
# Run 5 verification epochs with synthetic data on GPU:
python train.py --epochs 5 --val_interval 1 --batch_size 2 --amp_dtype bfloat16
```

Or generate synthetic NIfTI volumes explicitly:
```bash
python generate_sample_data.py --output_dir data/brain_mri --num_train 8 --num_val 2
```

---

### 4.3 Step 2: Production Training on NVIDIA H200
Place your NIfTI volumes in `data/brain_mri/` (or specify `--data_dir`):
```text
data/brain_mri/
├── imagesTr/          # Training T1w scans: sub-001_T1w.nii.gz, ...
├── labelsTr/          # Training masks:     sub-001_seg.nii.gz, ...
├── imagesVal/         # Validation scans:   sub-val-001_T1w.nii.gz, ...
└── labelsVal/         # Validation masks:   sub-val-001_seg.nii.gz, ...
```

Run production training:
```bash
python train.py \
    --data_dir /path/to/brain_mri \
    --output_dir outputs \
    --epochs 200 \
    --batch_size 4 \
    --lr 2e-4 \
    --roi_size 96 96 96 \
    --amp_dtype bfloat16 \
    --val_interval 2 \
    --num_workers 8 \
    --experiment_name "3D-Brain-SwinUNETR-Segmentation" \
    --run_name "H200-Production-Run"
```

---

### 4.4 Step 3: Experiment Tracking with MLflow
Launch the interactive MLflow dashboard to track real-time training loss, validation Dice scores, learning rate schedules, and GPU VRAM usage:

```bash
mlflow ui --port 5000
```
Open **`http://localhost:5000`** in your browser.

---

### 4.5 Step 4: Full-Volume Evaluation & NIfTI Mask Export
Run sliding window inference on any unseen patient volume and export the predicted 3D segmentation mask:

```bash
python evaluate.py \
    --checkpoint outputs/checkpoints/best_metric_model.pth \
    --input_nifti data/brain_mri/imagesVal/sub-val-001_T1w.nii.gz \
    --label_nifti data/brain_mri/labelsVal/sub-val-001_seg.nii.gz \
    --output_mask predicted_brain_seg.nii.gz
```
The exported `predicted_brain_seg.nii.gz` retains the exact physical affine matrix and spatial dimensions of the input MRI, allowing immediate visualization in **ITK-SNAP**, **3D Slicer**, or **FSLeyes**.

