#!/bin/bash
# =============================================================================
# 03_finetune.sh - Fine-tune LoRA adapters and a classification head
#
# Select the inputs with MODALITY. Each modality writes to its own directory,
# so runs do not overwrite each other:
#
#   MODALITY=dn     bash final_script/03_finetune.sh
#   MODALITY=dn+img bash final_script/03_finetune.sh
#
# Valid values: dn, img, rr, dn+img, dn+rr, img+rr, dn+img+rr
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
source "${SCRIPT_DIR}/modality.sh"
cd "${REPO_ROOT}"

OUTPUT_PATH="${TRAINED_MODELS_DIR}/${MODALITY}"

echo "=== Fine-tuning: modality=${MODALITY}, summary_type=${SUMMARY_TYPE} ==="
echo "    Data   : ${TRAIN_DATA_PATH}"
echo "    Output : ${OUTPUT_PATH}"
echo ""

python -m train.finetuning \
    --model_name_or_path "${MODEL_NAME}" \
    --output_path "${OUTPUT_PATH}" \
    --checkpoint_dir "${OUTPUT_PATH}" \
    --summary_type "${SUMMARY_TYPE}" \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --num_epochs "${NUM_EPOCHS}" \
    --lr "${LR}" \
    --seed "${SEED}" \
    --metadata_path "${DATASET_METADATA_PATH}" \
    --base_img_dir "${CXR_IMG_DIR}" \
    --base_rr_dir "${MIMIC_CXR_RR_DIR}" \
    --train_data_path "${TRAIN_DATA_PATH}" \
    --dev_data_path "${DEV_DATA_PATH}" \
    --test_data_path "${TEST_DATA_PATH}" \
    --train_metadata_image_path "${TRAIN_IMAGE_META_PATH}" \
    --dev_metadata_image_path "${DEV_IMAGE_META_PATH}" \
    --test_metadata_image_path "${TEST_IMAGE_META_PATH}" \
    "${MODALITY_FLAGS[@]}" \
    "$@"

echo ""
echo "=== Fine-tuning complete: ${OUTPUT_PATH} ==="
