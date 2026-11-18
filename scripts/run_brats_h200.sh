#!/usr/bin/env bash
# ==============================================================================
# run_brats_h200.sh — SOTA BraTS 3D Brain Tumour Training on NVIDIA H200 Cluster
# Framework : MONAI + Swin-UNETR (Pre-Trained on 5,050 volumes) + PyTorch DDP
# Target    : NVIDIA H200 Tensor Core GPUs (Hopper sm_90, 141 GB HBM3e)
# ==============================================================================

set -euo pipefail

# Choose GPU mode:
# Option 1: Single H200 (Default: GPU 0)
# Option 2: 4x H200 DDP (using GPUs 0, 5, 6, 7 which are completely free!)
NUM_GPUS="${1:-1}"
EPOCHS="${2:-100}"
BATCH_SIZE="${3:-4}"
DATA_DIR="${4:-/raid/Task01_BrainTumour}"
OUTPUT_DIR="${5:-./outputs_brats}"

export PYTHONUNBUFFERED="1"
export MLFLOW_ALLOW_FILE_STORE="true"

echo "======================================================================"
echo "  SOTA 3D SWIN-UNETR BraTS PIPELINE — NVIDIA H200 RUNNER"
echo "  Timestamp : $(date -Is)"
echo "  GPUs      : $NUM_GPUS H200 GPU(s)"
echo "  Epochs    : $EPOCHS"
echo "  Data Dir  : $DATA_DIR"
echo "======================================================================"

# Step 1: Download Real Clinical BraTS Dataset + Pretrained Encoder Weights
echo ""
echo "[Step 1/4] Ensuring BraTS dataset and 5,050-volume Pre-Trained Weights..."
python3 download_brats.py --data_dir "$DATA_DIR" --weights_dir pretrained

# Step 2: Launch Training (Single GPU or 4x H200 Distributed Data Parallel)
echo ""
echo "[Step 2/4] Launching High-Performance Swin-UNETR Training on H200..."
if [ "$NUM_GPUS" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES="0"
    python3 train_brats.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --roi_size 128 \
        --lr 1e-4 \
        --gpus 1 \
        --pretrained pretrained/model_swinvit.pt \
        --val_interval 2 \
        --num_workers 0
else
    # 4-GPU Distributed Data Parallel on verified free H200s (0, 5, 6, 7)
    export CUDA_VISIBLE_DEVICES="0,5,6,7"
    torchrun --nproc_per_node=4 train_brats.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --roi_size 128 \
        --lr 1e-4 \
        --gpus 4 \
        --pretrained pretrained/model_swinvit.pt \
        --val_interval 2 \
        --num_workers 0
fi

# Step 3: Run Full Clinical Evaluation on a Test Case
echo ""
echo "[Step 3/4] Evaluating Best Checkpoint on Validation Volume..."
SAMPLE_IMG=$(find "${DATA_DIR}/imagesTr" -name "*.nii.gz" 2>/dev/null | head -n 1)
if [ -z "$SAMPLE_IMG" ]; then
    SAMPLE_IMG=$(find "${DATA_DIR}" -name "*.nii.gz" 2>/dev/null | head -n 1)
fi
SAMPLE_NAME=$(basename "$SAMPLE_IMG")
SAMPLE_LBL="${DATA_DIR}/labelsTr/${SAMPLE_NAME}"

python3 evaluate_brats.py \
    --checkpoint "${OUTPUT_DIR}/checkpoints/best_brats_model.pth" \
    --image "$SAMPLE_IMG" \
    --label "$SAMPLE_LBL" \
    --output_mask "${OUTPUT_DIR}/pred_${SAMPLE_NAME}" \
    --roi_size 128

# Step 4: Generate All-In-One NiiVue Neurosurgical Studio
echo ""
echo "[Step 4/4] Generating All-In-One NiiVue Neurosurgical Studio (HTML)..."
python3 make_studio.py \
    "$SAMPLE_IMG" \
    "${OUTPUT_DIR}/pred_${SAMPLE_NAME}" \
    "${OUTPUT_DIR}/patient_neurosurgical_studio.html"

echo ""
echo "======================================================================"
echo "  PIPELINE COMPLETE!"
echo "  - Checkpoint  : ${OUTPUT_DIR}/checkpoints/best_brats_model.pth"
echo "  - NiiVue Studio: ${OUTPUT_DIR}/patient_neurosurgical_studio.html (open in browser)"
echo "======================================================================"

