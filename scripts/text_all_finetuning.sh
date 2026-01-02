#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

# 모델과 데이터 경로 설정
MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
OUTPUT_BASE="../finetuned_model"

for BATCH_SIZE in 16; do
  for SUMMARY_TYPE in plain; do
    # for COUNT in 1 2 3 4 5 6 7 8 9 10; do
    OUTPUT_PATH="${OUTPUT_BASE}/dn+rr/${SUMMARY_TYPE}"
    # python finetuning.py \
    #     --model_name_or_path "${MODEL_NAME}" \
    #     --output_path "${OUTPUT_PATH}" \
    #     --checkpoint_dir "${OUTPUT_PATH}" \
    #     --summary_type "${SUMMARY_TYPE}" \
    #     --batch_size ${BATCH_SIZE} \
    #     --num_epochs 30 \
    #     --gradient_accumulation_steps 2 \
    #     --use_discharge_note \
    #     --use_rad_report \
    #     --wandb \
        
    python inference.py \
        --model_name_or_path "${MODEL_NAME}" \
        --output_path "${OUTPUT_PATH}" \
        --checkpoint_dir "${OUTPUT_PATH}" \
        --summary_type "${SUMMARY_TYPE}" \
        --batch_size ${BATCH_SIZE} \
        --use_discharge_note \
        --use_rad_report \
        --use_pi \

    # done   
  done   
done   
