#!/bin/bash
# =============================================================================
# 02_summarize.sh - Summarize discharge notes with the base model
#
# Summarizes the original_note field of each split into the style named by
# SUMMARY_TYPE (plain, risk_factor, timeline, and the *_remove_cxr variants).
#
# Input:  ${DATASET_DIR}/{split}_summarization/{split}.jsonl
# Output: ${DATASET_DIR}/{split}_summarization/${SUMMARY_TYPE}_output.jsonl
#
# Steps 3 and 4 read that output file.
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${REPO_ROOT}"

for SPLIT in train dev test; do
    echo "=== [${SPLIT}] Summarizing discharge notes (summary_type=${SUMMARY_TYPE}) ==="

    OUTPUT_PATH="${DATASET_DIR}/${SPLIT}_summarization"

    python -m preprocessing.summarize_dn \
        --model_name_or_path "${MODEL_NAME}" \
        --set_name "${SPLIT}" \
        --summary_type "${SUMMARY_TYPE}" \
        --data_dir "${DATASET_DIR}" \
        --output_path "${OUTPUT_PATH}" \
        --metadata_path "${DATASET_METADATA_PATH}" \
        --base_img_dir "${CXR_IMG_DIR}" \
        --base_rr_dir "${MIMIC_CXR_RR_DIR}" \
        --train_metadata_image_path "${TRAIN_IMAGE_META_PATH}" \
        --dev_metadata_image_path "${DEV_IMAGE_META_PATH}" \
        --test_metadata_image_path "${TEST_IMAGE_META_PATH}" \
        --batch_size 1 \
        --summarize \
        --zeroshot \
        "$@"

    echo "  Saved: ${OUTPUT_PATH}/${SUMMARY_TYPE}_output.jsonl"
    echo ""
done

echo "=== Summarization complete ==="
