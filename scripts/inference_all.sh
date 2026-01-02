#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

# 모델과 데이터 경로 설정
MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
OUTPUT_BASE="../finetuned_model_new"


for BATCH_SIZE in 8; do
    for COUNT in 1 2 3 4 5; do
        for SUMMARY_TYPE in plain plain_remove_cxr risk_factor risk_factor_remove_cxr timeline timeline_remove_cxr; do

            OUTPUT_PATH="${OUTPUT_BASE}/dn/${SUMMARY_TYPE}"            
            python inference.py \
                --model_name_or_path "${MODEL_NAME}" \
                --output_path "${OUTPUT_PATH}" \
                --checkpoint_dir "${OUTPUT_PATH}" \
                --summary_type "${SUMMARY_TYPE}" \
                --batch_size ${BATCH_SIZE} \
                --use_discharge_note \

        done   
    done   
done   


for BATCH_SIZE in 8; do
    for COUNT in 1 2 3 4 5; do
        for SUMMARY_TYPE in plain plain_remove_cxr risk_factor risk_factor_remove_cxr timeline timeline_remove_cxr; do

            OUTPUT_PATH="${OUTPUT_BASE}/dn+rr/${SUMMARY_TYPE}"            
            python inference.py \
                --model_name_or_path "${MODEL_NAME}" \
                --output_path "${OUTPUT_PATH}" \
                --checkpoint_dir "${OUTPUT_PATH}" \
                --summary_type "${SUMMARY_TYPE}" \
                --batch_size ${BATCH_SIZE} \
                --use_discharge_note \
                --use_rad_report \

        done   
    done   
done   


for BATCH_SIZE in 8; do
    for COUNT in 1 2 3 4 5; do
        for SUMMARY_TYPE in plain plain_remove_cxr risk_factor risk_factor_remove_cxr timeline timeline_remove_cxr; do      
            OUTPUT_PATH="${OUTPUT_BASE}/dn+img/${SUMMARY_TYPE}"            
            python inference.py \
                --model_name_or_path "${MODEL_NAME}" \
                --output_path "${OUTPUT_PATH}" \
                --checkpoint_dir "${OUTPUT_PATH}" \
                --summary_type "${SUMMARY_TYPE}" \
                --batch_size ${BATCH_SIZE} \
                --use_discharge_note \
                --use_cxr_image

        done   
    done   
done   

# for BATCH_SIZE in 8; do
#     for COUNT in 1; do
#         for SUMMARY_TYPE in plain risk_factor timeline; do    
#             OUTPUT_PATH="${OUTPUT_BASE}/dn/${SUMMARY_TYPE}"            
#             python inference.py \
#                 --model_name_or_path "${MODEL_NAME}" \
#                 --output_path "${OUTPUT_PATH}" \
#                 --checkpoint_dir "${OUTPUT_PATH}" \
#                 --summary_type "${SUMMARY_TYPE}" \
#                 --batch_size ${BATCH_SIZE} \
#                 --use_discharge_note \
#                 --use_pi

#         done   
#     done   
# done   


# for BATCH_SIZE in 8; do
#     for COUNT in 1; do
#         for SUMMARY_TYPE in plain risk_factor timeline; do
#             OUTPUT_PATH="${OUTPUT_BASE}/dn+rr/${SUMMARY_TYPE}"            
#             python inference.py \
#                 --model_name_or_path "${MODEL_NAME}" \
#                 --output_path "${OUTPUT_PATH}" \
#                 --checkpoint_dir "${OUTPUT_PATH}" \
#                 --summary_type "${SUMMARY_TYPE}" \
#                 --batch_size ${BATCH_SIZE} \
#                 --use_discharge_note \
#                 --use_rad_report \
#                 --use_pi

#         done   
#     done   
# done   


# for BATCH_SIZE in 8; do
#     for COUNT in 1; do
#         for SUMMARY_TYPE in plain risk_factor timeline; do
#             OUTPUT_PATH="${OUTPUT_BASE}/dn+img/${SUMMARY_TYPE}"            
#             python inference.py \
#                 --model_name_or_path "${MODEL_NAME}" \
#                 --output_path "${OUTPUT_PATH}" \
#                 --checkpoint_dir "${OUTPUT_PATH}" \
#                 --summary_type "${SUMMARY_TYPE}" \
#                 --batch_size ${BATCH_SIZE} \
#                 --use_discharge_note \
#                 --use_cxr_image \
#                 --use_pi

#         done   
#     done   
# done   