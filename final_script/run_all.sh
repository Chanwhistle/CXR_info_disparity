#!/bin/bash
# =============================================================================
# run_all.sh — Run the full pipeline end to end
#
# Executes all steps in order:
#   preprocessing -> summarization -> fine-tuning -> inference
#
# Each step can also be run independently:
#   bash 01_preprocess.sh
#   bash 02_summarize.sh
#   bash 03_finetune.sh
#   bash 04_inference.sh
#
# Set paths and modality in config.sh and 03_finetune.sh before running.
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

if [ -z "${HF_TOKEN}" ]; then
    echo "[WARNING] HF_TOKEN is not set. Required for private model access."
    echo "          Run: export HF_TOKEN=hf_xxx  or set it in config.sh"
    echo ""
fi

log() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

log "STEP 1/4 — Preprocessing"
bash "${SCRIPT_DIR}/01_preprocess.sh"

log "STEP 2/4 — Discharge note summarization"
bash "${SCRIPT_DIR}/02_summarize.sh"

log "STEP 3/4 — Fine-tuning"
bash "${SCRIPT_DIR}/03_finetune.sh"

log "STEP 4/4 — Inference"
bash "${SCRIPT_DIR}/04_inference.sh"

log "Pipeline complete"
echo "Results: ${TRAINED_MODELS_DIR}/<modality>/score.txt"
