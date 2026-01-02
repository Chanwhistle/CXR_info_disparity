
#!/bin/bash
# 사용할 GPU 번호 설정
export CUDA_VISIBLE_DEVICES=0

# 모델과 데이터 경로 설정
MODEL_NAME="meta-llama/Llama-3.2-11B-Vision-Instruct"
OUTPUT_BASE="../zeroshot"

# Modified type 리스트
MODIFIED_TYPES=(
    "original_note"
)

# 반복문을 사용하여 실행
OUTPUT_PATH="${OUTPUT_BASE}/original_note"
python llm_zeroshot.py \
    --model_name_or_path "${MODEL_NAME}" \
    --output_path "${OUTPUT_PATH}" \
    --checkpoint_dir "${OUTPUT_PATH}" \
    --summary_type original_note \
    --batch_size 1 \
    --use_discharge_note \
    --zeroshot \
    
OUTPUT_PATH="${OUTPUT_BASE}/original_note"
python llm_zeroshot.py \
    --model_name_or_path "${MODEL_NAME}" \
    --output_path "${OUTPUT_PATH}" \
    --checkpoint_dir "${OUTPUT_PATH}" \
    --summary_type original_note \
    --batch_size 1 \
    --use_discharge_note \
    --use_rad_report \
    --zeroshot \

OUTPUT_PATH="${OUTPUT_BASE}/original_note"
python llm_zeroshot.py \
    --model_name_or_path "${MODEL_NAME}" \
    --output_path "${OUTPUT_PATH}" \
    --checkpoint_dir "${OUTPUT_PATH}" \
    --summary_type original_note \
    --batch_size 1 \
    --use_discharge_note \
    --use_cxr_image \
    --zeroshot \