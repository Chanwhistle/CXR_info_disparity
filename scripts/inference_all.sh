#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
# 모델과 데이터 경로 설정
MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
OUTPUT_BASE="../trained_models"
CXR_DIR="../physionet.org/files/mimic-cxr"

# for BATCH_SIZE in 1; do
#     for COUNT in 1; do
#         for SUMMARY_TYPE in plain plain_remove_cxr; do

#             OUTPUT_PATH="${OUTPUT_BASE}/dn"            
#             python inference.py \
#                 --model_name_or_path "${MODEL_NAME}" \
#                 --output_path "${OUTPUT_PATH}" \
#                 --checkpoint_dir "${OUTPUT_PATH}" \
#                 --summary_type "${SUMMARY_TYPE}" \
#                 --batch_size ${BATCH_SIZE} \
#                 --base_img_dir "${CXR_DIR}" \
#                 --base_rr_dir "${CXR_DIR}" \
#                 --use_discharge_note \

#         done   
#     done   
# done   


for BATCH_SIZE in 2; do
    for COUNT in 1; do
        for SUMMARY_TYPE in plain plain_remove_cxr; do

            OUTPUT_PATH="${OUTPUT_BASE}/dn+img"            
            python inference.py \
                --model_name_or_path "${MODEL_NAME}" \
                --output_path "${OUTPUT_PATH}" \
                --checkpoint_dir "${OUTPUT_PATH}" \
                --summary_type "${SUMMARY_TYPE}" \
                --batch_size ${BATCH_SIZE} \
                --base_img_dir "${CXR_DIR}" \
                --base_rr_dir "${CXR_DIR}" \
                --use_discharge_note \
                --use_cxr_image \
                --debug

        done   
    done   
done   


# for BATCH_SIZE in 1; do
#     for COUNT in 1; do
#         for SUMMARY_TYPE in plain plain_remove_cxr; do

#             OUTPUT_PATH="${OUTPUT_BASE}/dn+rr"            
#             python inference.py \
#                 --model_name_or_path "${MODEL_NAME}" \
#                 --output_path "${OUTPUT_PATH}" \
#                 --checkpoint_dir "${OUTPUT_PATH}" \
#                 --summary_type "${SUMMARY_TYPE}" \
#                 --batch_size ${BATCH_SIZE} \
#                 --base_img_dir "${CXR_DIR}" \
#                 --base_rr_dir "${CXR_DIR}" \
#                 --use_discharge_note \
#                 --use_rad_report \

#         done   
#     done   
# done   








# for BATCH_SIZE in 1; do
#     for COUNT in 1; do
#         for SUMMARY_TYPE in plain; do

#             OUTPUT_PATH="${OUTPUT_BASE}/dn"            
#             python inference.py \
#                 --model_name_or_path "${MODEL_NAME}" \
#                 --output_path "${OUTPUT_PATH}" \
#                 --checkpoint_dir "${OUTPUT_PATH}" \
#                 --summary_type "${SUMMARY_TYPE}" \
#                 --batch_size ${BATCH_SIZE} \
#                 --base_img_dir "${CXR_DIR}" \
#                 --base_rr_dir "${CXR_DIR}" \
#                 --use_discharge_note \
#                 --use_pi \

#         done   
#     done   
# done   


for BATCH_SIZE in 1; do
    for COUNT in 1; do
        for SUMMARY_TYPE in plain; do

            OUTPUT_PATH="${OUTPUT_BASE}/dn+img"            
            python inference.py \
                --model_name_or_path "${MODEL_NAME}" \
                --output_path "${OUTPUT_PATH}" \
                --checkpoint_dir "${OUTPUT_PATH}" \
                --summary_type "${SUMMARY_TYPE}" \
                --batch_size ${BATCH_SIZE} \
                --base_img_dir "${CXR_DIR}" \
                --base_rr_dir "${CXR_DIR}" \
                --use_discharge_note \
                --use_cxr_image \
                --use_pi \

        done   
    done   
done   


# for BATCH_SIZE in 1; do
#     for COUNT in 1; do
#         for SUMMARY_TYPE in plain; do

#             OUTPUT_PATH="${OUTPUT_BASE}/dn+rr"            
#             python inference.py \
#                 --model_name_or_path "${MODEL_NAME}" \
#                 --output_path "${OUTPUT_PATH}" \
#                 --checkpoint_dir "${OUTPUT_PATH}" \
#                 --summary_type "${SUMMARY_TYPE}" \
#                 --batch_size ${BATCH_SIZE} \
#                 --base_img_dir "${CXR_DIR}" \
#                 --base_rr_dir "${CXR_DIR}" \
#                 --use_discharge_note \
#                 --use_rad_report \
#                 --use_pi \

#         done   
#     done   
# done   