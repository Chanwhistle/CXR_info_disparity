#!/bin/bash
# Generate radiology reports using CheXagent 8B model
# This script processes CXR images from train/dev/test sets

export CUDA_VISIBLE_DEVICES=5

# Model and data path configuration
MODEL_NAME="StanfordAIMI/CheXagent-8b"
BASE_IMG_DIR="../saved_images"
BASE_RR_DIR="../physionet.org/files/mimic-cxr/2.1.0/files"
METADATA_PATH="../dataset/metadata.json"

# Metadata image paths
TRAIN_METADATA="../dataset/train_summarization/full-train-indent-images.json"
DEV_METADATA="../dataset/dev_summarization/full-dev-indent-images.json"
TEST_METADATA="../dataset/test_summarization/full-test-indent-images.json"

# Generation parameters
MAX_NEW_TOKENS=512
BATCH_SIZE=1

echo "=============================================="
echo "CheXagent Radiology Report Generation"
echo "=============================================="
echo "Model: ${MODEL_NAME}"
echo "Image Dir: ${BASE_IMG_DIR}"
echo "Report Dir: ${BASE_RR_DIR}"
echo "=============================================="

# Run generation for all splits
python generated_radiology_report.py \
    --model_name_or_path "${MODEL_NAME}" \
    --base_img_dir "${BASE_IMG_DIR}" \
    --base_rr_dir "${BASE_RR_DIR}" \
    --metadata_path "${METADATA_PATH}" \
    --train_metadata_image_path "${TRAIN_METADATA}" \
    --dev_metadata_image_path "${DEV_METADATA}" \
    --test_metadata_image_path "${TEST_METADATA}" \
    --splits train dev test \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --batch_size ${BATCH_SIZE} \
    --skip_existing

echo "=============================================="
echo "Generation Complete!"
echo "=============================================="
