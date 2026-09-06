#!/bin/bash
# =============================================================================
# 04_inference.sh - Score the test split
#
# Two modes, selected with RUN_MODE:
#   finetuned  load the checkpoint from 03_finetune.sh          (default)
#   zeroshot   run the base model with no fine-tuning
#
# Use the same MODALITY as in 03_finetune.sh:
#   MODALITY=dn RUN_MODE=finetuned bash final_script/04_inference.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
source "${SCRIPT_DIR}/modality.sh"
cd "${REPO_ROOT}"

RUN_MODE="${RUN_MODE:-finetuned}"
CHECKPOINT_PATH="${TRAINED_MODELS_DIR}/${MODALITY}"

COMMON_ARGS=(
    --model_name_or_path "${MODEL_NAME}"
    --summary_type "${SUMMARY_TYPE}"
    --batch_size 1
    --seed "${SEED}"
    --metadata_path "${DATASET_METADATA_PATH}"
    --base_img_dir "${CXR_IMG_DIR}"
    --base_rr_dir "${MIMIC_CXR_RR_DIR}"
    --dev_data_path "${DEV_DATA_PATH}"
    --test_data_path "${TEST_DATA_PATH}"
    --dev_metadata_image_path "${DEV_IMAGE_META_PATH}"
    --test_metadata_image_path "${TEST_IMAGE_META_PATH}"
)

case "${RUN_MODE}" in
    finetuned)
        echo "=== Fine-tuned inference: modality=${MODALITY} ==="
        echo "    Checkpoint: ${CHECKPOINT_PATH}"
        echo ""
        python -m eval.inference \
            "${COMMON_ARGS[@]}" \
            --output_path "${CHECKPOINT_PATH}" \
            --checkpoint_dir "${CHECKPOINT_PATH}" \
            "${MODALITY_FLAGS[@]}" \
            "$@"
        OUTPUT_PATH="${CHECKPOINT_PATH}"
        ;;
    zeroshot)
        OUTPUT_PATH="${TRAINED_MODELS_DIR}/zeroshot_${MODALITY}"
        echo "=== Zero-shot inference: modality=${MODALITY} ==="
        echo "    Output: ${OUTPUT_PATH}"
        echo ""
        python -m eval.llm_zeroshot \
            "${COMMON_ARGS[@]}" \
            --output_path "${OUTPUT_PATH}" \
            --zeroshot \
            "${MODALITY_FLAGS[@]}" \
            "$@"
        ;;
    *)
        echo "[ERROR] Unknown RUN_MODE: ${RUN_MODE}. Valid values: finetuned, zeroshot"
        exit 1
        ;;
esac

echo ""
echo "=== Inference complete. Results: ${OUTPUT_PATH}/score.txt ==="
