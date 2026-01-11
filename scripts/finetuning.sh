#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# 모델과 데이터 경로 설정
MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
OUTPUT_BASE="../trained_models"
CXR_DIR="../saved_images"
RR_DIR="../physionet.org/files/mimic-cxr/2.1.0/files"

for BATCH_SIZE in 4; do
  for SUMMARY_TYPE in plain; do
    for COUNT in 1; do
      for LR in 1e-4 5e-5 2e-5; do
        for HEAD_LR in 1e-3 5e-4; do
          OUTPUT_PATH="${OUTPUT_BASE}/img/${COUNT}_${BATCH_SIZE}_${LR}_${HEAD_LR}"
          python finetuning.py \
              --model_name_or_path ${MODEL_NAME} \
              --output_path ${OUTPUT_PATH} \
              --checkpoint_dir ${OUTPUT_PATH} \
              --summary_type ${SUMMARY_TYPE} \
              --batch_size ${BATCH_SIZE} \
              --base_img_dir ${CXR_DIR} \
              --base_rr_dir ${RR_DIR} \
              --lr ${LR} \
              --head_lr ${HEAD_LR} \
              --num_epochs 20 \
              --gradient_accumulation_steps 8 \
              --use_cxr_image \
                  
          python inference.py \
              --model_name_or_path ${MODEL_NAME} \
              --output_path ${OUTPUT_PATH} \
              --checkpoint_dir ${OUTPUT_PATH} \
              --summary_type ${SUMMARY_TYPE} \
              --batch_size ${BATCH_SIZE} \
              --base_img_dir ${CXR_DIR} \
              --base_rr_dir ${RR_DIR} \
              --use_cxr_image \

        done    
      done    
    done    
  done   
done   