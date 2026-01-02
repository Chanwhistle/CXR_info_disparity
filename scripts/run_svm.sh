#!/bin/bash

# CPU 전용 SVM 학습 스크립트
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 출력 경로 설정
OUTPUT_BASE="../finetuned_model_new"

echo "Starting SVM Baseline Training"
echo "==============================="

# SVM 하이퍼파라미터
# C: regularization 파라미터 (일반적으로 1.0이 기본값)
# kernel: 'linear' 또는 'rbf'
# SUMMARY_TYPE: 데이터 타입
KERNEL="rbf"

for SUMMARY_TYPE in plain plain_remove_cxr risk_factor risk_factor_remove_cxr timeline timeline_remove_cxr; do  
  OUTPUT_PATH="${OUTPUT_BASE}/svm/pi_${SUMMARY_TYPE}"
  
  echo ""
  echo "========================================"
  echo "Training SVM:"
  echo "  C: ${C_VALUE}"
  echo "  Kernel: ${KERNEL}"
  echo "  Summary: ${SUMMARY_TYPE}"
  echo "========================================"
  
  python svm.py \
      --output_path "${OUTPUT_PATH}" \
      --summary_type "${SUMMARY_TYPE}" \
      # --use_pi
  
  echo "Completed: ${SUMMARY_TYPE}"
  


  OUTPUT_PATH="${OUTPUT_BASE}/svm/pi_rr_${SUMMARY_TYPE}"
  
  echo ""
  echo "========================================"
  echo "Training SVM:"
  echo "  C: ${C_VALUE}"
  echo "  Kernel: ${KERNEL}"
  echo "  Summary: ${SUMMARY_TYPE}"
  echo "========================================"
  
  python svm.py \
      --output_path "${OUTPUT_PATH}" \
      --summary_type "${SUMMARY_TYPE}" \
      --use_rad_report \
      # --use_pi
  
  echo "Completed: ${SUMMARY_TYPE}"


  # 결과 출력
  if [ -f "${OUTPUT_PATH}/metrics.txt" ]; then
    echo ""
    grep "Test" "${OUTPUT_PATH}/metrics.txt"
    echo ""
  fi
  
done   

echo ""
echo "========================================"
echo "All experiments completed!"
echo "========================================"