#!/bin/bash
# Hyperparameter search for: run_experiment "dn+img" "plain" "false"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Every search run uses all four GPUs. Override with QWEN_GPU_IDS if needed.
GPU_IDS="${QWEN_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,3,5}}"
source "${REPO_ROOT}/final_script/config.sh"

MODEL_NAME="${QWEN_MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_DIR_NAME="${MODEL_NAME##*/}"
SEARCH_OUTPUT_DIR="${QWEN_SEARCH_OUTPUT_DIR:-${TRAINED_MODELS_DIR}/${MODEL_DIR_NAME}/dn_img_plain_hp_search}"

GLOBAL_BATCH_SIZES=(8 16 32)
LEARNING_RATES=(2e-5 1e-5 5e-6 2e-6 1e-6)

# Keep the per-device batch small for Qwen VL and obtain each requested global
# batch size through gradient accumulation.
MICRO_BATCH_SIZE="${QWEN_MICRO_BATCH_SIZE:-1}"

IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
NUM_GPUS="${#GPU_ID_ARRAY[@]}"

if [[ "${NUM_GPUS}" -ne 4 ]]; then
    echo "[ERROR] This search requires exactly 4 GPUs, but GPU_IDS=${GPU_IDS}"
    exit 1
fi

total_runs=$((${#GLOBAL_BATCH_SIZES[@]} * ${#LEARNING_RATES[@]}))
run_number=0

for global_batch_size in "${GLOBAL_BATCH_SIZES[@]}"; do
    denominator=$((MICRO_BATCH_SIZE * NUM_GPUS))
    if ((global_batch_size % denominator != 0)); then
        echo "[ERROR] Global batch size ${global_batch_size} is not divisible by"
        echo "        micro batch size ${MICRO_BATCH_SIZE} * ${NUM_GPUS} GPUs."
        exit 1
    fi
    grad_accum=$((global_batch_size / denominator))

    for lr in "${LEARNING_RATES[@]}"; do
        run_number=$((run_number + 1))
        output_path="${SEARCH_OUTPUT_DIR}/bs_${global_batch_size}_lr_${lr}"

        echo ""
        echo "=== Qwen hyperparameter search ${run_number}/${total_runs} ==="
        echo "GPUs:              ${GPU_IDS}"
        echo "Per-device batch:  ${MICRO_BATCH_SIZE}"
        echo "Gradient accum:    ${grad_accum}"
        echo "Global batch:      ${global_batch_size}"
        echo "Learning rate:     ${lr}"
        echo "Output:            ${output_path}"

        CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
        QWEN_MODEL_NAME="${MODEL_NAME}" \
        MODALITY="dn+img" \
        SUMMARY_TYPE="plain" \
        USE_PI="false" \
        BATCH_SIZE="${MICRO_BATCH_SIZE}" \
        GRAD_ACCUM="${grad_accum}" \
        LR="${lr}" \
        OUTPUT_PATH="${output_path}" \
        bash "${SCRIPT_DIR}/finetuning_qwen.sh"
    done
done

echo ""
echo "=== Qwen hyperparameter search complete: ${SEARCH_OUTPUT_DIR} ==="
