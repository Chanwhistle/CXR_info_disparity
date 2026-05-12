#!/bin/bash
# =============================================================================
# 01_preprocess.sh — Preprocessing pipeline
#
# Steps:
#   1. Add CXR image metadata to LCD Benchmark JSONs (train/dev/test)
#   2. Resize MIMIC-CXR JPG images to 512px
#   3. (Optional) Build HDF5 dataset for MultiChannel model
#
# Make sure to set paths in config.sh before running.
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

echo "=== [1/3] Adding CXR image metadata to JSON files ==="
cd "${PREPROCESSING_DIR}"

for SPLIT in train dev test; do
    INPUT_JSON="${LCD_JSON_DIR}/${SPLIT}.json"
    OUTPUT_JSON="${LCD_JSON_DIR}/${SPLIT}_with_images.json"

    if [ ! -f "${INPUT_JSON}" ]; then
        echo "[SKIP] ${INPUT_JSON} not found"
        continue
    fi

    echo "  ${SPLIT}: ${INPUT_JSON} -> ${OUTPUT_JSON}"
    python add_images_to_json.py \
        "${INPUT_JSON}" \
        "${MIMIC_CXR_JPG_DIR}" \
        "${OUTPUT_JSON}"
done

echo ""
echo "=== [2/3] Resizing MIMIC-CXR JPGs to 512px ==="
# Note: runs over the entire dataset, may take several hours
python resize_jpgs.py "${MIMIC_CXR_JPG_DIR}"

echo ""
echo "=== [3/3] (Optional) Building HDF5 dataset for MultiChannel model ==="
# Only needed when using train_mortality_xray.py with --model_type multichannel.
# Uncomment to run.

# for SPLIT in train dev test; do
#     INPUT_JSON="${LCD_JSON_DIR}/${SPLIT}_with_images.json"
#     OUTPUT_HDF5="${DATASET_DIR}/${SPLIT}_multichannel.hdf5"
#     echo "  ${SPLIT}: ${INPUT_JSON} -> ${OUTPUT_HDF5}"
#     python preprocess_mc_instances.py \
#         "${INPUT_JSON}" \
#         "${MIMIC_CXR_JPG_DIR}" \
#         "${OUTPUT_HDF5}"
# done

echo ""
echo "=== Preprocessing complete ==="
