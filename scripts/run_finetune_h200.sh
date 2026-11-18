#!/usr/bin/env bash
# ==============================================================================
# run_finetune_h200.sh — High-Speed ET Fine-Tuning on NVIDIA H200 Cluster
# Framework : MONAI + Swin-UNETR (Starting from best_brats_model.pth)
# Target    : NVIDIA H200 Tensor Core GPUs (Hopper sm_90)
# ==============================================================================

set -euo pipefail

NUM_GPUS="${1:-4}"
EPOCHS="${2:-15}"
BATCH_SIZE="${3:-4}"
LR="${4:-1e-4}"

DATA_DIR="${5:-/raid/Task01_BrainTumour}"
if [ ! -d "$DATA_DIR" ]; then
    DATA_DIR="./data/Task01_BrainTumour"
fi

OUTPUT_DIR="${6:-./outputs_brats}"
CHECKPOINT="${OUTPUT_DIR}/checkpoints/best_brats_model.pth"
if [ ! -f "$CHECKPOINT" ]; then
    CHECKPOINT="checkpoints/best_brats_model.pth"
fi

export PYTHONUNBUFFERED="1"
export MLFLOW_ALLOW_FILE_STORE="true"

echo "======================================================================"
echo "  3D SWIN-UNETR BraTS ENHANCING TUMOR (ET) FINE-TUNER"
echo "  Timestamp  : $(date -Is)"
echo "  GPUs       : $NUM_GPUS H200 GPU(s)"
echo "  Epochs     : $EPOCHS (Estimated ~10-12 mins on 4x H200)"
echo "  Data Dir   : $DATA_DIR"
echo "  Checkpoint : $CHECKPOINT"
echo "  ET Label   : 3 (MSD Task01 Decathlon Specification)"
echo "======================================================================"

if [ "$NUM_GPUS" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES="0"
    python3 train_brats.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --resume "$CHECKPOINT" \
        --epochs "$EPOCHS" \
        --warmup_epochs 1 \
        --batch_size "$BATCH_SIZE" \
        --roi_size 128 \
        --lr "$LR" \
        --gpus 1 \
        --et_label 3 \
        --val_interval 2 \
        --num_workers 0
else
    # 4-GPU Distributed Data Parallel on GPUs 0, 5, 6, 7
    export CUDA_VISIBLE_DEVICES="0,5,6,7"
    torchrun --nproc_per_node=4 train_brats.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --resume "$CHECKPOINT" \
        --epochs "$EPOCHS" \
        --warmup_epochs 1 \
        --batch_size "$BATCH_SIZE" \
        --roi_size 128 \
        --lr "$LR" \
        --gpus 4 \
        --et_label 3 \
        --val_interval 2 \
        --num_workers 4
fi

echo ""
echo "[Evaluation] Benchmarking Updated Model on Validation Volume..."
SAMPLE_IMG=$(find "${DATA_DIR}/imagesTr" -name "*460*.nii.gz" 2>/dev/null | head -n 1)
if [ -z "$SAMPLE_IMG" ]; then
    SAMPLE_IMG=$(find "${DATA_DIR}/imagesTr" -name "*.nii.gz" 2>/dev/null | head -n 1)
fi
SAMPLE_NAME=$(basename "$SAMPLE_IMG")
SAMPLE_LBL="${DATA_DIR}/labelsTr/${SAMPLE_NAME}"

python3 evaluate_brats.py \
    --checkpoint "${OUTPUT_DIR}/checkpoints/best_brats_model.pth" \
    --image "$SAMPLE_IMG" \
    --label "$SAMPLE_LBL" \
    --output_mask "${OUTPUT_DIR}/pred_${SAMPLE_NAME}" \
    --roi_size 128

echo ""
echo "[Studio] Regenerating Interactive 3D Neurosurgical Studio..."
python3 make_studio.py \
    "$SAMPLE_IMG" \
    "${OUTPUT_DIR}/pred_${SAMPLE_NAME}" \
    "${OUTPUT_DIR}/patient_neurosurgical_studio.html"

echo ""
echo "======================================================================"
echo "  ET FINE-TUNING COMPLETE!"
echo "  - New Best Checkpoint : ${OUTPUT_DIR}/checkpoints/best_brats_model.pth"
echo "  - New NiiVue Studio   : ${OUTPUT_DIR}/patient_neurosurgical_studio.html"
echo "======================================================================"

