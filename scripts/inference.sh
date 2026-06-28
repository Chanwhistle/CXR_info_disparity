#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

# 모델과 데이터 경로 설정
MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
OUTPUT_BASE="${REPO_ROOT}/trained_models"
CXR_DIR="${REPO_ROOT}/saved_images_560"
RR_DIR="${REPO_ROOT}/physionet.org/files/mimic-cxr/2.1.0/files"
DATASET_DIR="${REPO_ROOT}/dataset"
METADATA_PATH="${DATASET_DIR}/metadata.json"


# for BATCH_SIZE in 1; do
#     for COUNT in 1; do
#         for SUMMARY_TYPE in plain; do

#             OUTPUT_PATH="${OUTPUT_BASE}/rr"            
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
#     for COUNT in 1; do
#         for SUMMARY_TYPE in plain; do

#             OUTPUT_PATH="${OUTPUT_BASE}/img"            
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



# for BATCH_SIZE in 1; do
#     for COUNT in 1; do
#         for SUMMARY_TYPE in plain; do

#             OUTPUT_PATH="${OUTPUT_BASE}/tmp2"            
#             python inference.py \
#                 --model_name_or_path "${MODEL_NAME}" \
#                 --output_path "${OUTPUT_PATH}" \
#                 --checkpoint_dir "${OUTPUT_PATH}" \
#                 --summary_type "${SUMMARY_TYPE}" \
#                 --batch_size ${BATCH_SIZE} \
#                 --base_img_dir "${CXR_DIR}" \
#                 --base_rr_dir "${RR_DIR}" \
#                 --use_discharge_note \

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
#                 --base_rr_dir "${RR_DIR}" \
#                 --use_discharge_note \
#                 --use_pi \

#         done   
#     done   
# done   



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
#                 --base_rr_dir "${RR_DIR}" \
#                 --use_discharge_note \
#                 --use_rad_report \

#         done   
#     done   
# done   


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
#                 --base_rr_dir "${RR_DIR}" \
#                 --use_discharge_note \
#                 --use_rad_report \
#                 --use_pi \

#         done   
#     done   
# done   



# for BATCH_SIZE in 1; do
#     for COUNT in 1; do
#         for SUMMARY_TYPE in plain plain_remove_cxr; do

#             OUTPUT_PATH="${OUTPUT_BASE}/dn+img"            
#             python inference.py \
#                 --model_name_or_path "${MODEL_NAME}" \
#                 --output_path "${OUTPUT_PATH}" \
#                 --checkpoint_dir "${OUTPUT_PATH}" \
#                 --summary_type "${SUMMARY_TYPE}" \
#                 --batch_size ${BATCH_SIZE} \
#                 --base_img_dir "${CXR_DIR}" \
#                 --base_rr_dir "${RR_DIR}" \
#                 --use_discharge_note \
#                 --use_cxr_image \

#         done   
#     done   
# done   


for BATCH_SIZE in 1; do
    for COUNT in 1; do
        for SUMMARY_TYPE in plain; do

            OUTPUT_PATH="${OUTPUT_BASE}/dn+img"            
            python eval/inference.py \
                --model_name_or_path "${MODEL_NAME}" \
                --output_path "${OUTPUT_PATH}" \
                --checkpoint_dir "${OUTPUT_PATH}" \
                --summary_type "${SUMMARY_TYPE}" \
                --batch_size ${BATCH_SIZE} \
                --base_img_dir "${CXR_DIR}" \
                --base_rr_dir "${RR_DIR}" \
                --metadata_path "${METADATA_PATH}" \
                --test_data_path "${DATASET_DIR}/test_summarization/total_output.jsonl" \
                --test_metadata_image_path "${DATASET_DIR}/test_summarization/full-test-indent-images.json" \
                --use_discharge_note \
                --use_cxr_image \

        done   
    done   
done   



# for BATCH_SIZE in 1; do
#     for COUNT in 1; do
#         for SUMMARY_TYPE in plain plain_remove_cxr; do

#             OUTPUT_PATH="${OUTPUT_BASE}/dn+rr_gen"            
#             python inference.py \
#                 --model_name_or_path "${MODEL_NAME}" \
#                 --output_path "${OUTPUT_PATH}" \
#                 --checkpoint_dir "${OUTPUT_PATH}" \
#                 --summary_type "${SUMMARY_TYPE}" \
#                 --batch_size ${BATCH_SIZE} \
#                 --base_img_dir "${CXR_DIR}" \
#                 --base_rr_dir "${RR_DIR}" \
#                 --use_discharge_note \
#                 --use_generated_rad_report \

#         done   
#     done   
# done   


# for BATCH_SIZE in 1; do
#     for COUNT in 1; do
#         for SUMMARY_TYPE in plain; do

#             OUTPUT_PATH="${OUTPUT_BASE}/dn+rr_gen"            
#             python inference.py \
#                 --model_name_or_path "${MODEL_NAME}" \
#                 --output_path "${OUTPUT_PATH}" \
#                 --checkpoint_dir "${OUTPUT_PATH}" \
#                 --summary_type "${SUMMARY_TYPE}" \
#                 --batch_size ${BATCH_SIZE} \
#                 --base_img_dir "${CXR_DIR}" \
#                 --base_rr_dir "${RR_DIR}" \
#                 --use_discharge_note \
#                 --use_generated_rad_report \
#                 --use_pi \

#         done   
#     done   
# done   
