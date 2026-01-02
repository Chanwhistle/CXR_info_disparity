#!/bin/bash
export CUDA_VISIBLE_DEVICES=5

# 모델과 데이터 경로 설정
MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
OUTPUT_BASE="../dataset"

SUMMARY_TYPES=(
    "plain"
    # "plain_remove_cxr"
    # "risk_factor"
    # "risk_factor_remove_cxr"
    # "timeline"
    # "timeline_remove_cxr"
)

SET_NAME=(
    "train"
    "dev"
    "test"
)

# 반복문을 사용하여 실행
for TYPE in "${SUMMARY_TYPES[@]}"; do
    for SET in "${SET_NAME[@]}"; do
        echo "${TYPE} ${SET} Inference!"
        OUTPUT_PATH="${OUTPUT_BASE}/${SET}_summarization2"
        python summarize_dn.py \
            --model_name_or_path "${MODEL_NAME}" \
            --output_path "${OUTPUT_PATH}" \
            --summary_type "${TYPE}" \
            --batch_size 1 \
            --set_name "${SET}" \
            --summarize \
            --zeroshot \

    done
done
