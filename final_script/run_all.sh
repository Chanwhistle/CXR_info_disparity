#!/bin/bash
# =============================================================================
# run_all.sh - Run the full pipeline end to end
#
#   preprocessing -> summarization -> fine-tuning -> inference
#
# Each step can also be run on its own:
#   bash final_script/01_preprocess.sh
#   bash final_script/02_summarize.sh
#   MODALITY=dn bash final_script/03_finetune.sh
#   MODALITY=dn bash final_script/04_inference.sh
#
# Set paths in config.sh first. Pick inputs with MODALITY:
#   MODALITY=dn+img bash final_script/run_all.sh
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

export MODALITY="${MODALITY:-dn}"

if [ -z "${HF_TOKEN}" ]; then
    echo "[WARNING] HF_TOKEN is not set. Gated models such as Llama-3.2 will fail to download."
    echo "          Run: export HF_TOKEN=hf_xxx"
    echo ""
fi

log() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

log "STEP 1/4 - Preprocessing"
bash "${SCRIPT_DIR}/01_preprocess.sh"

log "STEP 2/4 - Discharge note summarization"
bash "${SCRIPT_DIR}/02_summarize.sh"

log "STEP 3/4 - Fine-tuning (modality=${MODALITY})"
bash "${SCRIPT_DIR}/03_finetune.sh"

log "STEP 4/4 - Inference (modality=${MODALITY})"
bash "${SCRIPT_DIR}/04_inference.sh"

log "Pipeline complete"
echo "Results: ${TRAINED_MODELS_DIR}/${MODALITY}/score.txt"
