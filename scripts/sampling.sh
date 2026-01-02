#!/bin/bash
export HF_HOME=/hdd2/chanhwi/huggingface_cache
export TRANSFORMERS_CACHE=/hdd2/chanhwi/huggingface_cache
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=6
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

# 모델과 데이터 경로 설정
MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
RESULT="../inference_result"
OUTPUT_BASE="../finetuned_model"

for BATCH_SIZE in 8; do
  for SUMMARY_TYPE in plain; do
      OUTPUT_PATH="${OUTPUT_BASE}/dn/${SUMMARY_TYPE}"
            
      python inference.py \
          --model_name_or_path "${MODEL_NAME}" \
          --output_path "${RESULT}/dn" \
          --checkpoint_dir "${OUTPUT_PATH}" \
          --summary_type "${SUMMARY_TYPE}" \
          --batch_size ${BATCH_SIZE} \
          --use_discharge_note \
          --use_pi \

  done   
done   
