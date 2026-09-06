#!/bin/bash
# =============================================================================
# 01_preprocess.sh - Link LCD splits to CXRs and write files for steps 2-4
#
# Expects LCD Benchmark train/dev/test.json and metadata.json (from
# https://github.com/Machine-Learning-for-Medical-Language/long-clinical-doc).
# Then selects one CXR per admission, resizes it, and writes JSONL + image metadata.
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"
cd "${REPO_ROOT}"

echo "=== [1/2] Checking LCD Benchmark split files ==="

missing=0
for name in train.json dev.json test.json metadata.json; do
    path="${LCD_JSON_DIR}/${name}"
    if [ ! -f "${path}" ]; then
        echo "[ERROR] Missing ${path}"
        missing=1
    fi
done
if [ "${missing}" -ne 0 ]; then
    echo "        Build those files with the LCD Benchmark repo:"
    echo "        https://github.com/Machine-Learning-for-Medical-Language/long-clinical-doc"
    exit 1
fi
echo "  Found splits in ${LCD_JSON_DIR}"

echo ""
echo "=== [2/2] Building JSONL, image metadata, and resized CXRs ==="
# Writes:
#   ${DATASET_DIR}/{split}_summarization/{split}.jsonl
#   ${DATASET_DIR}/{split}_summarization/full-{split}-indent-images.json
#   ${DATASET_DIR}/metadata.json
#   ${CXR_IMG_DIR}/{split}/*_560_resized.jpg

python -m preprocessing.build_lcd_cxr_dataset \
    --lcd_dir "${LCD_JSON_DIR}" \
    --metadata_path "${LCD_METADATA_PATH}" \
    --mimic_iv_dir "${MIMIC_IV_DIR}" \
    --mimic_cxr_jpg_dir "${MIMIC_CXR_JPG_DIR}" \
    --dataset_dir "${DATASET_DIR}" \
    --image_output_dir "${CXR_IMG_DIR}" \
    --task_name "${TASK_NAME}"

echo ""
echo "=== Preprocessing complete ==="
echo "    Dataset : ${DATASET_DIR}"
echo "    Images  : ${CXR_IMG_DIR}"
