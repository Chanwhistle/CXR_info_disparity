#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

# 모델과 데이터 경로 설정
MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
OUTPUT_BASE="../finetuned_model"

for BATCH_SIZE in 8; do
  for SUMMARY_TYPE in plain; do  
    for LORA_SETTING in 1 2 3; do  
      for COUNT in 1 2 3 4 5 6 7 8 9 10; do
        OUTPUT_PATH="${OUTPUT_BASE}/lora_ablation/${SUMMARY_TYPE}_${LORA_SETTING}"
        # torchrun --nproc_per_node=$NUM_GPUS --master_port=29506  finetuning.py \
        #     --model_name_or_path "${MODEL_NAME}" \
        #     --output_path "${OUTPUT_PATH}" \
        #     --summary_type "${SUMMARY_TYPE}" \
        #     --batch_size ${BATCH_SIZE} \
        #     --lr 1e-6 \
        #     --num_epochs 30 \
        #     --gradient_accumulation_steps 2 \
        #     --use_cxr_image \
        #     --use_discharge_note \
        #     --lora_setting ${LORA_SETTING} \
        #     --wandb
            
        python inference.py \
            --model_name_or_path "${MODEL_NAME}" \
            --output_path "${OUTPUT_PATH}" \
            --checkpoint_dir "${OUTPUT_PATH}" \
            --summary_type "${SUMMARY_TYPE}" \
            --batch_size 1 \
            --use_cxr_image \
            --use_discharge_note \
            --lora_setting ${LORA_SETTING} \
            --d

      done   
    done   
  done   
done   