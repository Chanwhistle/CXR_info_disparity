#!/bin/bash
# Temporary Qwen zero-shot evaluation for DN and DN+CXR.
# Remove this file together with tmp/qwen_zeroshot/ when no longer needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/final_script/config.sh"

cd "${REPO_ROOT}"

MODEL_NAME="${QWEN_MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_DIR_NAME="${MODEL_DIR_NAME:-${MODEL_NAME##*/}}"
ZEROSHOT_OUTPUT_DIR="${ZEROSHOT_OUTPUT_DIR:-${REPO_ROOT}/tmp/qwen_zeroshot/results/${MODEL_DIR_NAME}}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"

if ! python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "[ERROR] CUDA is not available inside this environment."
    exit 1
fi

COMMON_ARGS=(
    --model_name_or_path "${MODEL_NAME}"
    --summary_type "${SUMMARY_TYPE}"
    --test_data_path "${DATASET_DIR}/test_summarization/total_output.jsonl"
    --metadata_path "${METADATA_PATH}"
    --test_metadata_image_path "${DATASET_DIR}/test_summarization/full-test-indent-images.json"
    --base_img_dir "${CXR_IMG_DIR}"
    --base_rr_dir "${MIMIC_CXR_RR_DIR}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --seed "${SEED}"
)

if [ -n "${MAX_SAMPLES}" ]; then
    COMMON_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

if [ "${LOAD_IN_4BIT:-0}" = "1" ]; then
    COMMON_ARGS+=(--load_in_4bit)
fi

for MODALITY in dn dn+img; do
    OUTPUT_PATH="${ZEROSHOT_OUTPUT_DIR}/${MODALITY}"
    echo "=== Qwen zero-shot: modality=${MODALITY}, output=${OUTPUT_PATH} ==="
    python -m tmp.qwen_zeroshot.evaluate \
        "${COMMON_ARGS[@]}" \
        --modality "${MODALITY}" \
        --output_path "${OUTPUT_PATH}"
done

echo "=== Qwen zero-shot evaluation complete: ${ZEROSHOT_OUTPUT_DIR} ==="
