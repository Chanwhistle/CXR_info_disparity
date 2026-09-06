#!/bin/bash
# =============================================================================
# modality.sh - Translate a MODALITY name into model flags
#
# Sourced by 03_finetune.sh and 04_inference.sh so both steps always agree on
# which inputs the model sees. Select one with the MODALITY variable:
#
#   MODALITY=dn+img bash final_script/03_finetune.sh
# =============================================================================

MODALITY="${MODALITY:-dn}"

case "${MODALITY}" in
    dn)        MODALITY_FLAGS=(--use_discharge_note) ;;
    img)       MODALITY_FLAGS=(--use_cxr_image) ;;
    rr)        MODALITY_FLAGS=(--use_rad_report) ;;
    dn+img)    MODALITY_FLAGS=(--use_discharge_note --use_cxr_image) ;;
    dn+rr)     MODALITY_FLAGS=(--use_discharge_note --use_rad_report) ;;
    img+rr)    MODALITY_FLAGS=(--use_cxr_image --use_rad_report) ;;
    dn+img+rr) MODALITY_FLAGS=(--use_discharge_note --use_cxr_image --use_rad_report) ;;
    *)
        echo "[ERROR] Unknown MODALITY: ${MODALITY}"
        echo "        Valid values: dn, img, rr, dn+img, dn+rr, img+rr, dn+img+rr"
        exit 1
        ;;
esac
