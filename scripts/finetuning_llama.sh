#!/bin/bash
# Fine-tune and evaluate the main experiment with Llama Vision.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/final_script/config.sh"

cd "${REPO_ROOT}"

# Keep this script compatible with older config.sh versions that do not
# define the metadata mapper explicitly.
METADATA_PATH="${METADATA_PATH:-${REPO_ROOT}/dataset/metadata.json}"

MODEL_NAME="${LLAMA_MODEL_NAME:-meta-llama/Llama-3.2-11B-Vision-Instruct}"
MODEL_FAMILY="llama"
MODEL_DIR_NAME="${MODEL_DIR_NAME:-${MODEL_NAME##*/}}"
MODALITY="${MODALITY:-dn}"
OUTPUT_PATH="${OUTPUT_PATH:-${TRAINED_MODELS_DIR}/${MODEL_DIR_NAME}/${MODALITY}}"
USE_PI="${USE_PI:-false}"

case "${MODALITY}" in
    dn)     MODALITY_FLAGS=(--use_discharge_note) ;;
    img)    MODALITY_FLAGS=(--use_cxr_image) ;;
    rr)     MODALITY_FLAGS=(--use_rad_report) ;;
    dn+img) MODALITY_FLAGS=(--use_discharge_note --use_cxr_image) ;;
    dn+rr)  MODALITY_FLAGS=(--use_discharge_note --use_rad_report) ;;
    *)
        echo "[ERROR] Unsupported MODALITY: ${MODALITY}"
        echo "        Choose one of: dn, img, rr, dn+img, dn+rr"
        exit 1
        ;;
esac

EXTRA_FLAGS=()
case "${USE_PI,,}" in
    true|1|yes) EXTRA_FLAGS+=(--use_pi) ;;
    false|0|no) ;;
    *)
        echo "[ERROR] USE_PI must be true or false: ${USE_PI}"
        exit 1
        ;;
esac

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${#GPU_IDS[@]}"

if ! python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "[ERROR] CUDA is not available inside this environment."
    echo "        Recreate the Docker Compose container with GPU access:"
    echo "        docker compose down && docker compose up -d --build cxr-app"
    echo "        Then verify: docker compose exec cxr-app nvidia-smi"
    exit 1
fi

COMMON_ARGS=(
    --model_name_or_path "${MODEL_NAME}"
    --model_family "${MODEL_FAMILY}"
    --summary_type "${SUMMARY_TYPE}"
    --metadata_path "${METADATA_PATH}"
    --base_img_dir "${CXR_IMG_DIR}"
    --base_rr_dir "${MIMIC_CXR_RR_DIR}"
    --train_data_path "${DATASET_DIR}/train_summarization/total_output.jsonl"
    --dev_data_path "${DATASET_DIR}/dev_summarization/total_output.jsonl"
    --test_data_path "${DATASET_DIR}/test_summarization/total_output.jsonl"
    --train_metadata_image_path "${DATASET_DIR}/train_summarization/full-train-indent-images.json"
    --dev_metadata_image_path "${DATASET_DIR}/dev_summarization/full-dev-indent-images.json"
    --test_metadata_image_path "${DATASET_DIR}/test_summarization/full-test-indent-images.json"
)

REQUIRED_FILES=(
    "${METADATA_PATH}"
    "${DATASET_DIR}/train_summarization/total_output.jsonl"
    "${DATASET_DIR}/dev_summarization/total_output.jsonl"
    "${DATASET_DIR}/test_summarization/total_output.jsonl"
    "${DATASET_DIR}/train_summarization/full-train-indent-images.json"
    "${DATASET_DIR}/dev_summarization/full-dev-indent-images.json"
    "${DATASET_DIR}/test_summarization/full-test-indent-images.json"
)

for path in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Required file not found: ${path}"
        echo "        Set DATASET_DIR or METADATA_PATH to the prepared dataset location."
        exit 1
    fi
done

echo "=== Llama main experiment ==="
echo "Model:      ${MODEL_NAME}"
echo "Modality:   ${MODALITY}"
echo "Use PI:     ${USE_PI}"
echo "GPUs:       ${CUDA_VISIBLE_DEVICES} (${NUM_GPUS})"
echo "Output:     ${OUTPUT_PATH}"
echo "Dataset:    ${DATASET_DIR}"
echo "Metadata:   ${METADATA_PATH}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" -m train.finetuning \
    "${COMMON_ARGS[@]}" \
    --output_path "${OUTPUT_PATH}" \
    --checkpoint_dir "${OUTPUT_PATH}" \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --num_epochs "${NUM_EPOCHS}" \
    --lr "${LR}" \
    --seed "${SEED}" \
    "${MODALITY_FLAGS[@]}" \
    "${EXTRA_FLAGS[@]}"

checkpoint_count=0
for checkpoint_dir in "${OUTPUT_PATH}"/checkpoint-*; do
    if [ ! -d "${checkpoint_dir}" ]; then
        continue
    fi

    checkpoint_count=$((checkpoint_count + 1))
    checkpoint_name="$(basename "${checkpoint_dir}")"
    result_path="${OUTPUT_PATH}/test_results/${checkpoint_name}"
    echo "=== Testing Llama checkpoint: ${checkpoint_name} ==="

    python -m eval.inference \
        "${COMMON_ARGS[@]}" \
        --output_path "${result_path}" \
        --checkpoint_dir "${checkpoint_dir}" \
        --batch_size 1 \
        --seed "${SEED}" \
        "${MODALITY_FLAGS[@]}" \
        "${EXTRA_FLAGS[@]}"
done

if [ "${checkpoint_count}" -eq 0 ]; then
    echo "[ERROR] No saved checkpoints found in: ${OUTPUT_PATH}"
    exit 1
fi

echo "=== Llama main experiment complete: ${OUTPUT_PATH} ==="
