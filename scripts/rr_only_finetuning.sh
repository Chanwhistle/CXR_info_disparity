#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

# 모델과 데이터 경로 설정
MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
OUTPUT_BASE="../finetuned_model_new"

for COUNT in 10 100; do
  for LR in 5e-5; do
    for BATCH_SIZE in 8; do
      for SUMMARY_TYPE in plain; do  
        OUTPUT_PATH="${OUTPUT_BASE}/rr/seed_${COUNT}"
        # python finetuning.py \
        #     --model_name_or_path "${MODEL_NAME}" \
        #     --output_path "${OUTPUT_PATH}" \
        #     --checkpoint_dir "${OUTPUT_PATH}" \
        #     --summary_type "${SUMMARY_TYPE}" \
        #     --batch_size ${BATCH_SIZE} \
        #     --lr ${LR} \
        #     --num_epochs 30 \
        #     --gradient_accumulation_steps 2 \
        #     --use_rad_report \
        #     --wandb \
              
        python inference.py \
            --model_name_or_path "${MODEL_NAME}" \
            --output_path "${OUTPUT_PATH}" \
            --checkpoint_dir "${OUTPUT_PATH}" \
            --summary_type "${SUMMARY_TYPE}" \
            --batch_size 1 \
            --use_rad_report \
            --seed "${COUNT}"

      done   
    done   
  done   
done   
