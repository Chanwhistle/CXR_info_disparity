#!/bin/bash
# =============================================================================
# 03_finetune.sh — Fine-tune the Llama Vision model
#
# Uncomment exactly one MODALITY block below, then run this script.
# Each modality saves its checkpoint to a separate directory.
#
# Available modalities:
#   dn      discharge note only
#   img     CXR image only
#   rr      radiology report only
#   dn+img  discharge note + CXR image
#   dn+rr   discharge note + radiology report
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

cd "${EXPERIMENTS_DIR}"

# ---------------------------------------------------------------------------
# Select one modality (uncomment one block)
# ---------------------------------------------------------------------------

# --- [1] Discharge note only ---
MODALITY="dn"
MODALITY_FLAGS="--use_discharge_note"

# --- [2] CXR image only ---
# MODALITY="img"
# MODALITY_FLAGS="--use_cxr_image"

# --- [3] Radiology report only ---
# MODALITY="rr"
# MODALITY_FLAGS="--use_rad_report"

# --- [4] Discharge note + CXR image ---
# MODALITY="dn+img"
# MODALITY_FLAGS="--use_discharge_note --use_cxr_image"

# --- [5] Discharge note + radiology report ---
# MODALITY="dn+rr"
# MODALITY_FLAGS="--use_discharge_note --use_rad_report"

# ---------------------------------------------------------------------------
OUTPUT_PATH="${TRAINED_MODELS_DIR}/${MODALITY}"

echo "=== Fine-tuning: modality=${MODALITY}, summary_type=${SUMMARY_TYPE} ==="
echo "    Output: ${OUTPUT_PATH}"
echo ""

python finetuning.py \
    --model_name_or_path "${MODEL_NAME}" \
    --output_path "${OUTPUT_PATH}" \
    --checkpoint_dir "${OUTPUT_PATH}" \
    --summary_type "${SUMMARY_TYPE}" \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --num_epochs "${NUM_EPOCHS}" \
    --lr "${LR}" \
    --head_lr "${HEAD_LR}" \
    --seed "${SEED}" \
    --base_img_dir "${CXR_IMG_DIR}" \
    --base_rr_dir "${MIMIC_CXR_RR_DIR}" \
    --train_data_path "${DATASET_DIR}/train_summarization/total_output.jsonl" \
    --dev_data_path "${DATASET_DIR}/dev_summarization/total_output.jsonl" \
    --test_data_path "${DATASET_DIR}/test_summarization/total_output.jsonl" \
    --train_metadata_image_path "${DATASET_DIR}/train_summarization/full-train-indent-images.json" \
    --dev_metadata_image_path "${DATASET_DIR}/dev_summarization/full-dev-indent-images.json" \
    --test_metadata_image_path "${DATASET_DIR}/test_summarization/full-test-indent-images.json" \
    ${MODALITY_FLAGS}

echo ""
echo "=== Fine-tuning complete: ${OUTPUT_PATH} ==="
