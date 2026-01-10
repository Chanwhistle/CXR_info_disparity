#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
# 모델과 데이터 경로 설정
MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
OUTPUT_BASE="../trained_models"
CXR_DIR="../saved_images"
RR_DIR="../physionet.org/files/mimic-cxr/2.1.0/files"


# for BATCH_SIZE in 1; do
#     for COUNT in 1 2 3 4 5; do
#         for SUMMARY_TYPE in plain; do

#             OUTPUT_PATH="${OUTPUT_BASE}/rr/${COUNT}"            
#             python inference.py \
#                 --model_name_or_path "${MODEL_NAME}" \
#                 --output_path "${OUTPUT_PATH}" \
#                 --checkpoint_dir "${OUTPUT_PATH}" \
#                 --summary_type "${SUMMARY_TYPE}" \
#                 --batch_size ${BATCH_SIZE} \
#                 --base_img_dir "${CXR_DIR}" \
#                 --base_rr_dir "${RR_DIR}" \
#                 --use_rad_report \
                
#         done   
#     done   
# done   


# for BATCH_SIZE in 1; do
#     for COUNT in 1 2 3 4 5; do
#         for SUMMARY_TYPE in plain; do

#             OUTPUT_PATH="${OUTPUT_BASE}/img/${COUNT}"            
#             python inference.py \
#                 --model_name_or_path "${MODEL_NAME}" \
#                 --output_path "${OUTPUT_PATH}" \
#                 --checkpoint_dir "${OUTPUT_PATH}" \
#                 --summary_type "${SUMMARY_TYPE}" \
#                 --batch_size ${BATCH_SIZE} \
#                 --base_img_dir "${CXR_DIR}" \
#                 --base_rr_dir "${RR_DIR}" \
#                 --use_cxr_image \

#         done   
#     done   
# done   


for BATCH_SIZE in 1; do
    for COUNT in 10; do
        for SUMMARY_TYPE in plain; do

            OUTPUT_PATH="${OUTPUT_BASE}/dn+img/${COUNT}"            
            python inference.py \
                --model_name_or_path "${MODEL_NAME}" \
                --output_path "${OUTPUT_PATH}" \
                --checkpoint_dir "${OUTPUT_PATH}" \
                --summary_type "${SUMMARY_TYPE}" \
                --batch_size ${BATCH_SIZE} \
                --base_img_dir "${CXR_DIR}" \
                --base_rr_dir "${RR_DIR}" \
                --use_discharge_note \
                --use_cxr_image \

        done   
    done   
done   
