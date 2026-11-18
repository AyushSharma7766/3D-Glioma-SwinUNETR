#!/usr/bin/env bash
# ==============================================================================
# SOTA 3D Swin-UNETR Master Production Pipeline on NVIDIA H200 Cluster
# Location   : 100% on /raid (27 TB Free Storage, Zero Permission Headaches)
# Dataset    : 484 Real Clinical BraTS Multi-Modal Cases (MSD Task01)
# Hardware   : 4x NVIDIA H200 GPUs (Hopper sm_90, 141 GB HBM3e each)
# Training   : PyTorch DDP · BF16 AMP · 128³ Patches · MONAI 5,050 Pre-Trained
# Deliverable: 100% Self-Contained NiiVue Dual-Sync Neurosurgical Studio HTML
# ==============================================================================

set -euo pipefail

DATA_DIR="/raid/Task01_BrainTumour"
PROJECT_DIR="/raid/3D-Brain-SwinUNETR-Segmentation"
OUTPUT_DIR="${PROJECT_DIR}/outputs_brats"
PRETRAINED="${PROJECT_DIR}/pretrained/model_swinvit.pt"

export PYTHONUNBUFFERED="1"
export MLFLOW_ALLOW_FILE_STORE="true"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

echo "======================================================================"
echo "  SOTA 3D SWIN-UNETR ALL-IN-RAID PIPELINE — 4x NVIDIA H200"
echo "  Timestamp : $(date -Is)"
echo "  Data Dir  : $DATA_DIR"
echo "  Project   : $PROJECT_DIR"
echo "  Output Dir: $OUTPUT_DIR"
echo "======================================================================"

mkdir -p "${OUTPUT_DIR}/checkpoints"

# Phase 1: Immediate Real Brain NiiVue Studio (Ready in 5 seconds!)
echo ""
echo "[Phase 1/3] Generating Immediate Real Brain Neurosurgical Studio..."
SAMPLE_IMG=$(find "${DATA_DIR}/imagesTr" -name "*.nii.gz" 2>/dev/null | head -n 1)
SAMPLE_NAME=$(basename "$SAMPLE_IMG")
SAMPLE_LBL="${DATA_DIR}/labelsTr/${SAMPLE_NAME}"

echo "  Patient Scan : $SAMPLE_IMG"
echo "  Clinical GT  : $SAMPLE_LBL"

python3 "${PROJECT_DIR}/make_studio.py" \
    "$SAMPLE_IMG" \
    "$SAMPLE_LBL" \
    "${PROJECT_DIR}/patient_neurosurgical_studio.html"

echo "  -> Instant Real Studio Ready: ${PROJECT_DIR}/patient_neurosurgical_studio.html"

# Phase 2: Distributed High-Performance Swin-UNETR Training on 4x H200
echo ""
echo "[Phase 2/3] Launching Distributed 4x H200 Training on 484 Real Patients..."
torchrun --nproc_per_node=4 "${PROJECT_DIR}/train_brats.py" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 100 \
    --batch_size 4 \
    --roi_size 128 \
    --lr 1e-4 \
    --gpus 4 \
    --pretrained "$PRETRAINED" \
    --val_interval 2 \
    --num_workers 4

# Phase 3: Sliding-Window Gaussian Inference on Clinical Case
echo ""
echo "[Phase 3/3] Evaluating Best Model & Generating AI-Predicted Studio..."
BEST_CKPT="${OUTPUT_DIR}/checkpoints/best_brats_model.pth"

python3 "${PROJECT_DIR}/evaluate_brats.py" \
    --checkpoint "$BEST_CKPT" \
    --image "$SAMPLE_IMG" \
    --label "$SAMPLE_LBL" \
    --output_mask "${OUTPUT_DIR}/pred_${SAMPLE_NAME}" \
    --roi_size 128

# Final Polish: Re-generate Studio with AI Swin-UNETR Predicted Tumor
python3 "${PROJECT_DIR}/make_studio.py" \
    "$SAMPLE_IMG" \
    "${OUTPUT_DIR}/pred_${SAMPLE_NAME}" \
    "${PROJECT_DIR}/patient_neurosurgical_studio.html"

chmod -R 777 "${PROJECT_DIR}" 2>/dev/null || true

echo ""
echo "======================================================================"
echo "  MASTER PIPELINE 100% COMPLETE!"
echo "  - Best Model Checkpoint : ${BEST_CKPT}"
echo "  - AI Prediction Mask    : ${OUTPUT_DIR}/pred_${SAMPLE_NAME}"
echo "  - NiiVue 3D Studio HTML : ${PROJECT_DIR}/patient_neurosurgical_studio.html"
echo "======================================================================"
