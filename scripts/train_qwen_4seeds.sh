#!/bin/bash
# Train and evaluate Qwen on four modality settings using four additional seeds.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Each run uses all listed GPUs through torchrun. Runs are executed sequentially.
GPU_IDS="${QWEN_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-2,4,5,6}}"
source "${REPO_ROOT}/final_script/config.sh"

MODEL_NAME="${QWEN_MODEL_NAME:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_DIR_NAME="${MODEL_NAME##*/}"

# Keep these runs separate from the existing seed-42 experiments.
SEED_RUNS_ROOT="${QWEN_SEED_RUNS_ROOT:-${REPO_ROOT}/trained_models_qwen_4seeds}"
OUTPUT_ROOT="${SEED_RUNS_ROOT}/${MODEL_DIR_NAME}"

# Override with a space-separated list, for example:
# QWEN_SEEDS="11 22 33 44" bash scripts/train_qwen_4seeds.sh
read -r -a SEEDS <<< "${QWEN_SEEDS:-77 777 7777}"
MODALITIES=("dn+img")

declare -A SEEN_SEEDS=()
for seed in "${SEEDS[@]}"; do
    if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Every seed must be a non-negative integer: ${seed}"
        exit 1
    fi
    if [[ -n "${SEEN_SEEDS[${seed}]:-}" ]]; then
        echo "[ERROR] Duplicate seed: ${seed}"
        exit 1
    fi
    SEEN_SEEDS["${seed}"]=1
done

total_runs=$((${#MODALITIES[@]} * ${#SEEDS[@]}))
run_number=0

echo "=== Qwen four-seed experiment plan ==="
echo "Model:       ${MODEL_NAME}"
echo "Modalities:  ${MODALITIES[*]}"
echo "Seeds:       ${SEEDS[*]}"
echo "GPUs/run:    ${GPU_IDS}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Runs:        ${total_runs} (sequential)"
echo "Config:      batch=${BATCH_SIZE}, grad_accum=${GRAD_ACCUM}, epochs=${NUM_EPOCHS}, lr=${LR}"

for modality in "${MODALITIES[@]}"; do
    experiment_name="${modality//+/_}_plain_pi"

    for seed in "${SEEDS[@]}"; do
        run_number=$((run_number + 1))
        output_path="${OUTPUT_ROOT}/${experiment_name}/seed_${seed}"

        echo ""
        echo "=== Qwen seed run ${run_number}/${total_runs}: ${experiment_name}, seed=${seed} ==="
        echo "Output: ${output_path}"

        if [[ -e "${output_path}" && "${QWEN_ALLOW_EXISTING_OUTPUT:-false}" != "true" ]]; then
            echo "[ERROR] Output path already exists: ${output_path}"
            echo "        Set QWEN_ALLOW_EXISTING_OUTPUT=true only when resuming intentionally."
            exit 1
        fi

        if [[ "${QWEN_DRY_RUN:-false}" == "true" ]]; then
            continue
        fi

        CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
        QWEN_MODEL_NAME="${MODEL_NAME}" \
        MODALITY="${modality}" \
        SUMMARY_TYPE="plain" \
        USE_PI="true" \
        SEED="${seed}" \
        OUTPUT_PATH="${output_path}" \
        bash "${SCRIPT_DIR}/finetuning_qwen.sh"
    done
done

echo ""
if [[ "${QWEN_DRY_RUN:-false}" == "true" ]]; then
    echo "=== Qwen four-seed dry run complete ==="
else
    echo "=== Qwen four-seed experiments complete: ${OUTPUT_ROOT} ==="
fi
