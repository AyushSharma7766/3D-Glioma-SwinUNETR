# 3D Swin-UNETR Model Checkpoints

This folder contains the trained PyTorch weights for the 3D Swin-UNETR brain tumor segmentation network.

## Checkpoint Details

* **File:** `best_brats_model.pth`
* **File Size:** ~256.4 MB (62,194,595 trainable parameters)
* **Architecture:** 3D Swin Transformer U-Net (`feature_size=48`, depths `(2, 2, 2, 2)`, heads `(3, 6, 12, 24)`)
* **Training Setup:** Distributed Data Parallel (DDP) across 4× NVIDIA H200 (141 GB HBM3e) GPUs
* **Input Channels:** 4 (T1, T1ce, T2, FLAIR)
* **Output Channels:** 3 overlapping regions (Tumor Core, Whole Tumor, Enhancing Tumor)

## Quantitative Benchmark Metrics

| Region | Anatomical Meaning | Validation Dice Score |
| :--- | :--- | :--- |
| **Whole Tumor (WT)** | Necrotic Core + Enhancing Rim + Vasogenic Edema | **90.03%** |
| **Tumor Core (TC)** | Necrotic Core + Enhancing Rim (Surgical target) | **82.27%** |
| **Enhancing Tumor (ET)** | Viable, hyper-vascularized proliferating rim | **79.10%** |
| **Mean Composite Dice** | Macro-average across all 3 subregions | **83.75%** |

## GitHub Large File Storage (LFS) / Download Note
Because GitHub enforces a **100 MB hard file limit**, `best_brats_model.pth` (256 MB) is excluded from standard `git push` via `.gitignore`.

To use the model:
1. Place your trained `best_brats_model.pth` directly in this `checkpoints/` directory.
2. If hosting on GitHub, attach `best_brats_model.pth` as an asset to your repository's **GitHub Releases** page or use **Git LFS**:
   ```bash
   git lfs install
   git lfs track "*.pth"
   git add .gitattributes
   git add checkpoints/best_brats_model.pth
   ```
