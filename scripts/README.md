# High-Performance Cluster Shell Scripts

This directory contains shell execution scripts used to train, fine-tune, and benchmark 3D Swin-UNETR on GPU clusters.

## Hardware Optimization

All scripts are configured for:
* **Hardware:** 4× NVIDIA H200 (141 GB HBM3e) Hopper GPUs
* **Distributed Backend:** PyTorch `torchrun` Distributed Data Parallel (DDP) with `nccl`
* **Precision:** Automatic Mixed Precision (AMP) with FP16/BF16

## Scripts Overview

* `run_brats_h200.sh`: Distributed baseline training script on 4× H200 GPUs.
* `run_finetune_h200.sh`: Targeted fine-tuning script with corrected MSD Task01 label mapping (`et_label=3`) for 15 epochs.
* `run_master_pipeline.sh`: End-to-end master pipeline script executing data verification, training, quantitative evaluation, and web studio generation.
