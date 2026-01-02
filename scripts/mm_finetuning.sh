#!/bin/bash
export HF_HOME=/hdd2/chanhwi/huggingface_cache
export TRANSFORMERS_CACHE=/hdd2/chanhwi/huggingface_cache
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=5
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

# 모델과 데이터 경로 설정
MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
OUTPUT_BASE="../finetuned_model"

for BATCH_SIZE in 4; do
  for SUMMARY_TYPE in plain; do
    for LR in 1e-6; do
      for COUNT in 1 2 3; do
      # for COUNT in 1 2 3 4 5 6 7 8 9 10; do
        OUTPUT_PATH="${OUTPUT_BASE}/dn+img/${SUMMARY_TYPE}_lora_${LR}"
        # python finetuning.py \
        #     --model_name_or_path "${MODEL_NAME}" \
        #     --output_path "${OUTPUT_PATH}" \
        #     --checkpoint_dir "${OUTPUT_PATH}" \
        #     --summary_type "${SUMMARY_TYPE}" \
        #     --batch_size ${BATCH_SIZE} \
        #     --lr ${LR} \
        #     --num_epochs 30 \
        #     --gradient_accumulation_steps 4 \
        #     --use_cxr_image \
        #     --use_discharge_note \
        #     --lora_setting 2 \
        #     --wandb
                
        python inference.py \
              --model_name_or_path "${MODEL_NAME}" \
              --output_path "${OUTPUT_PATH}" \
              --checkpoint_dir "${OUTPUT_PATH}" \
              --summary_type "${SUMMARY_TYPE}" \
              --batch_size ${BATCH_SIZE} \
              --use_cxr_image \
              --use_discharge_note \
              --lora_setting 2 

      done    
    done    
  done   
done   