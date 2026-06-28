#!/bin/bash
# =============================================================================
# 02_summarize.sh — Discharge note summarization using Llama
#
# Summarizes the original_note field in each split's JSONL using Llama,
# and saves the output to {SPLIT}_summarization/{SUMMARY_TYPE}_output.jsonl.
#
# Input:  ${DATASET_DIR}/{SPLIT}_summarization/{SPLIT}.jsonl
# Output: ${DATASET_DIR}/{SPLIT}_summarization/{SUMMARY_TYPE}_output.jsonl
#
# The finetuning and inference scripts read from these output files.
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

cd "${EXPERIMENTS_DIR}"

for SPLIT in train dev test; do
    echo "=== [${SPLIT}] Summarizing discharge notes (summary_type=${SUMMARY_TYPE}) ==="

    OUTPUT_PATH="${DATASET_DIR}/${SPLIT}_summarization"

    python summarize_dn.py \
        --model_name_or_path "${MODEL_NAME}" \
        --set_name "${SPLIT}" \
        --summary_type "${SUMMARY_TYPE}" \
        --data_dir "${DATASET_DIR}" \
        --output_path "${OUTPUT_PATH}" \
        --base_img_dir "${CXR_IMG_DIR}" \
        --base_rr_dir "${MIMIC_CXR_RR_DIR}" \
        --batch_size 1 \
        --summarize \
        --zeroshot

    echo "  Saved: ${OUTPUT_PATH}/${SUMMARY_TYPE}_output.jsonl"
    echo ""
done

echo "=== Summarization complete ==="
