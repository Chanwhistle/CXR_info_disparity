#!/bin/bash
# Run Llama experiments with and without radiology information in the DN summary.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Capture the requested Llama devices before config.sh assigns its single-GPU
# default. LLAMA_GPU_IDS can be used without changing other pipeline scripts.
GPU_IDS="${LLAMA_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-2,4,5,6}}"
source "${REPO_ROOT}/final_script/config.sh"

MODEL_NAME="${LLAMA_MODEL_NAME:-meta-llama/Llama-3.2-11B-Vision-Instruct}"
MODEL_DIR_NAME="${MODEL_NAME##*/}"

run_experiment() {
    local modality="$1"
    local summary_type="$2"
    local use_pi="$3"
    local setting_name="${summary_type}"
    if [[ "${use_pi}" == "true" ]]; the
        setting_name="${summary_type}_pi"
    fi
    local experiment_name="${modality//+/_}_${setting_name}"
    local output_path="${TRAINED_MODELS_DIR}/${MODEL_DIR_NAME}/${experiment_name}"

    echo ""
    echo "=== Llama experiment: ${experiment_name} ==="

    CUDA_VISIBLE_DEVICES="${GPU_IDS}" \
    MODALITY="${modality}" \
    SUMMARY_TYPE="${summary_type}" \
    USE_PI="${use_pi}" \
    OUTPUT_PATH="${output_path}" \
    bash "${SCRIPT_DIR}/finetuning_llama.sh"
}

# Standard discharge-note summaries.
run_experiment "dn+img" "plain" "false"
# run_experiment "dn" "plain" "false"
# run_experiment "dn+rr" "plain" "false"

# # Discharge-note summaries generated without chest imaging information.
# run_experiment "dn" "plain_remove_cxr" "false"
# run_experiment "dn+rr" "plain_remove_cxr" "false"
# run_experiment "dn+img" "plain_remove_cxr" "false"

# # Standard summaries with age and race.
# run_experiment "dn" "plain" "true"
# run_experiment "dn+rr" "plain" "true"
# run_experiment "dn+img" "plain" "true"
