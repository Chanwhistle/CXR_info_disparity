MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
OUTPUT_PATH="/ssd1/chanhwi/long-clinical-doc/hoonick/output_for_hoonick"

for SUMMARY_TYPE in plain_remove_cxr risk_factor_remove_cxr timeline_remove_cxr; do
          
    python sample_maker.py \
        --model_name_or_path "${MODEL_NAME}" \
        --output_path "${OUTPUT_PATH}" \
        --checkpoint_dir "${OUTPUT_PATH}" \
        --summary_type "${SUMMARY_TYPE}" \
        --use_discharge_note \

done   