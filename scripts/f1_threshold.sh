#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

# 모델과 데이터 경로 설정
MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
OUTPUT_BASE="../finetuned_model"

for LR in 2e-6; do
  for BATCH_SIZE in 4; do
    for SUMMARY_TYPE in plain; do 
      OUTPUT_PATH=/ssd1/chanhwi/long-clinical-doc/finetuned_model/Final_model/dn+img/1e-5_merged_ve_full        
      python inference.py \
          --model_name_or_path "${MODEL_NAME}" \
          --output_path "${OUTPUT_PATH}" \
          --checkpoint_dir "${OUTPUT_PATH}" \
          --summary_type "${SUMMARY_TYPE}" \
          --batch_size 8 \
          --use_cxr_image \
          --use_discharge_note \
          --debug

    done   
  done   
done   
