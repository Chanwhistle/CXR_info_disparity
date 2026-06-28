#!/bin/bash
# =============================================================================
# 04_inference.sh — Run inference with a fine-tuned or zero-shot model
#
# Two modes:
#   A) Fine-tuned checkpoint inference on dev/test (inference.py)
#   B) Zero-shot inference without fine-tuning (llm_zeroshot.py)
#
# Set MODALITY to match the modality used in 03_finetune.sh.
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

cd "${EXPERIMENTS_DIR}"

# ---------------------------------------------------------------------------
# Select the same modality as used in 03_finetune.sh
# ---------------------------------------------------------------------------

MODALITY="dn"
MODALITY_FLAGS="--use_discharge_note"

# MODALITY="img"
# MODALITY_FLAGS="--use_cxr_image"

# MODALITY="rr"
# MODALITY_FLAGS="--use_rad_report"

# MODALITY="dn+img"
# MODALITY_FLAGS="--use_discharge_note --use_cxr_image"

# MODALITY="dn+rr"
# MODALITY_FLAGS="--use_discharge_note --use_rad_report"

# ---------------------------------------------------------------------------
CHECKPOINT_PATH="${TRAINED_MODELS_DIR}/${MODALITY}"
OUTPUT_PATH="${CHECKPOINT_PATH}"

COMMON_ARGS=(
    --model_name_or_path "${MODEL_NAME}"
    --summary_type "${SUMMARY_TYPE}"
    --batch_size 1
    --seed "${SEED}"
    --base_img_dir "${CXR_IMG_DIR}"
    --base_rr_dir "${MIMIC_CXR_RR_DIR}"
    --dev_data_path "${DATASET_DIR}/dev_summarization/total_output.jsonl"
    --test_data_path "${DATASET_DIR}/test_summarization/total_output.jsonl"
    --dev_metadata_image_path "${DATASET_DIR}/dev_summarization/full-dev-indent-images.json"
    --test_metadata_image_path "${DATASET_DIR}/test_summarization/full-test-indent-images.json"
)

# ---------------------------------------------------------------------------
# A) Fine-tuned checkpoint inference
# ---------------------------------------------------------------------------
echo "=== [A] Fine-tuned inference: modality=${MODALITY} ==="
echo "    Checkpoint: ${CHECKPOINT_PATH}"
echo ""

python inference.py \
    "${COMMON_ARGS[@]}" \
    --output_path "${OUTPUT_PATH}" \
    --checkpoint_dir "${CHECKPOINT_PATH}" \
    ${MODALITY_FLAGS}

echo ""

# ---------------------------------------------------------------------------
# B) Zero-shot inference (uncomment to run separately)
# ---------------------------------------------------------------------------
# echo "=== [B] Zero-shot inference: modality=${MODALITY} ==="
# ZEROSHOT_OUTPUT="${TRAINED_MODELS_DIR}/zeroshot_${MODALITY}"
#
# python llm_zeroshot.py \
#     "${COMMON_ARGS[@]}" \
#     --output_path "${ZEROSHOT_OUTPUT}" \
#     --zeroshot \
#     ${MODALITY_FLAGS}

echo "=== Inference complete. Results: ${OUTPUT_PATH}/score.txt ==="
