#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

# 모델과 데이터 경로 설정
MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
OUTPUT_BASE="../finetuned_model_new"

for LR in 5e-5 2e-5 1e-5; do
  for BATCH_SIZE in 8; do
    for LORA_SETTING in 2; do
      for SEED in 22; do
        OUTPUT_PATH="${OUTPUT_BASE}/img/seed_${SEED}_LR_${LR}"
        python finetuning.py \
            --model_name_or_path "${MODEL_NAME}" \
            --output_path "${OUTPUT_PATH}" \
            --checkpoint_dir "${OUTPUT_PATH}" \
            --batch_size ${BATCH_SIZE} \
            --lr ${LR} \
            --num_epochs 30 \
            --gradient_accumulation_steps 2 \
            --lora_setting ${LORA_SETTING} \
            --use_cxr_image \
            --seed "${SEED}" \
            # --wandb \
              
        python inference.py \
            --model_name_or_path "${MODEL_NAME}" \
            --output_path "${OUTPUT_PATH}" \
            --checkpoint_dir "${OUTPUT_PATH}" \
            --batch_size 1 \
            --lora_setting ${LORA_SETTING} \
            --use_cxr_image \
            --seed "${SEED}"

      done   
    done   
  done   
done   
