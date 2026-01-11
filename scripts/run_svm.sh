#!/bin/bash

# CPU 전용 SVM 학습 스크립트
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
CXR_DIR="../saved_images"
RR_DIR="../physionet.org/files/mimic-cxr/2.1.0/files"

# 출력 경로 설정
OUTPUT_BASE="../finetuned_model_new"

echo "Starting SVM Training"
echo "====================="
echo "Experiments: DN and DN+RR for plain, plain_remove_cxr, plain+pi"
echo ""

# SVM 하이퍼파라미터
KERNEL="rbf"

# 결과 저장을 위한 배열
declare -A RESULTS_AUROC
declare -A RESULTS_AUPRC

# Summary types: plain, plain_remove_cxr, plain_pi
# Input types: DN (no rad_report), DN+RR (with rad_report)

for SUMMARY_TYPE in plain plain_remove_cxr; do
  # 1. DN only (no radiology report)
  OUTPUT_PATH="${OUTPUT_BASE}/svm/dn_${SUMMARY_TYPE}"
  
  echo ""
  echo "========================================"
  echo "Training SVM: DN + ${SUMMARY_TYPE}"
  echo "  Kernel: ${KERNEL}"
  echo "  Summary: ${SUMMARY_TYPE}"
  echo "  Input: DN only"
  echo "========================================"
  
  python svm.py \
    --output_path "${OUTPUT_PATH}" \
    --summary_type "${SUMMARY_TYPE}" \
    --base_img_dir "${CXR_DIR}" \
    --base_rr_dir "${RR_DIR}" \
  
  # 결과 추출
  if [ -f "${OUTPUT_PATH}/metrics.txt" ]; then
    TEST_AUROC=$(grep "Test" "${OUTPUT_PATH}/metrics.txt" | grep "auroc" | awk '{print $2}')
    TEST_AUPRC=$(grep "Test" "${OUTPUT_PATH}/metrics.txt" | grep "auprc" | awk '{print $2}')
    RESULTS_AUROC["dn_${SUMMARY_TYPE}"]="${TEST_AUROC}"
    RESULTS_AUPRC["dn_${SUMMARY_TYPE}"]="${TEST_AUPRC}"
    echo "  Test AUROC: ${TEST_AUROC}"
    echo "  Test AUPRC: ${TEST_AUPRC}"
  fi
  
  # 2. DN + RR (with radiology report)
  OUTPUT_PATH="${OUTPUT_BASE}/svm/dn_rr_${SUMMARY_TYPE}"
  
  echo ""
  echo "========================================"
  echo "Training SVM: DN+RR + ${SUMMARY_TYPE}"
  echo "  Kernel: ${KERNEL}"
  echo "  Summary: ${SUMMARY_TYPE}"
  echo "  Input: DN + Radiology Report"
  echo "========================================"
  
  python svm.py \
    --output_path "${OUTPUT_PATH}" \
    --summary_type "${SUMMARY_TYPE}" \
    --use_rad_report \
    --base_img_dir "${CXR_DIR}" \
    --base_rr_dir "${RR_DIR}" \
  
  # 결과 추출
  if [ -f "${OUTPUT_PATH}/metrics.txt" ]; then
    TEST_AUROC=$(grep "Test" "${OUTPUT_PATH}/metrics.txt" | grep "auroc" | awk '{print $2}')
    TEST_AUPRC=$(grep "Test" "${OUTPUT_PATH}/metrics.txt" | grep "auprc" | awk '{print $2}')
    RESULTS_AUROC["dn_rr_${SUMMARY_TYPE}"]="${TEST_AUROC}"
    RESULTS_AUPRC["dn_rr_${SUMMARY_TYPE}"]="${TEST_AUPRC}"
    echo "  Test AUROC: ${TEST_AUROC}"
    echo "  Test AUPRC: ${TEST_AUPRC}"
  fi
done

# 3. plain + pi (with personal information)
SUMMARY_TYPE="plain"

# 3-1. DN + PI
OUTPUT_PATH="${OUTPUT_BASE}/svm/dn_${SUMMARY_TYPE}_pi"

echo ""
echo "========================================"
echo "Training SVM: DN + ${SUMMARY_TYPE} + PI"
echo "  Kernel: ${KERNEL}"
echo "  Summary: ${SUMMARY_TYPE}"
echo "  Input: DN + Personal Information"
echo "========================================"

python svm.py \
  --output_path "${OUTPUT_PATH}" \
  --summary_type "${SUMMARY_TYPE}" \
  --use_pi \
  --base_img_dir "${CXR_DIR}" \
  --base_rr_dir "${RR_DIR}" \

# 결과 추출
if [ -f "${OUTPUT_PATH}/metrics.txt" ]; then
  TEST_AUROC=$(grep "Test" "${OUTPUT_PATH}/metrics.txt" | grep "auroc" | awk '{print $2}')
  TEST_AUPRC=$(grep "Test" "${OUTPUT_PATH}/metrics.txt" | grep "auprc" | awk '{print $2}')
  RESULTS_AUROC["dn_${SUMMARY_TYPE}_pi"]="${TEST_AUROC}"
  RESULTS_AUPRC["dn_${SUMMARY_TYPE}_pi"]="${TEST_AUPRC}"
  echo "  Test AUROC: ${TEST_AUROC}"
  echo "  Test AUPRC: ${TEST_AUPRC}"
fi

# 3-2. DN + RR + PI
OUTPUT_PATH="${OUTPUT_BASE}/svm/dn_rr_${SUMMARY_TYPE}_pi"

echo ""
echo "========================================"
echo "Training SVM: DN+RR + ${SUMMARY_TYPE} + PI"
echo "  Kernel: ${KERNEL}"
echo "  Summary: ${SUMMARY_TYPE}"
echo "  Input: DN + Radiology Report + Personal Information"
echo "========================================"

python svm.py \
  --output_path "${OUTPUT_PATH}" \
  --summary_type "${SUMMARY_TYPE}" \
  --use_rad_report \
  --use_pi \
  --base_img_dir "${CXR_DIR}" \
  --base_rr_dir "${RR_DIR}" \

# 결과 추출
if [ -f "${OUTPUT_PATH}/metrics.txt" ]; then
  TEST_AUROC=$(grep "Test" "${OUTPUT_PATH}/metrics.txt" | grep "auroc" | awk '{print $2}')
  TEST_AUPRC=$(grep "Test" "${OUTPUT_PATH}/metrics.txt" | grep "auprc" | awk '{print $2}')
  RESULTS_AUROC["dn_rr_${SUMMARY_TYPE}_pi"]="${TEST_AUROC}"
  RESULTS_AUPRC["dn_rr_${SUMMARY_TYPE}_pi"]="${TEST_AUPRC}"
  echo "  Test AUROC: ${TEST_AUROC}"
  echo "  Test AUPRC: ${TEST_AUPRC}"
fi

# 최종 결과 요약 출력
echo ""
echo "========================================"
echo "FINAL RESULTS SUMMARY"
echo "========================================"
echo ""
echo "Test AUROC Results:"
echo "-------------------"
printf "%-30s %10s\n" "Configuration" "AUROC"
echo "----------------------------------------"
printf "%-30s %10s\n" "DN + plain" "${RESULTS_AUROC[dn_plain]}"
printf "%-30s %10s\n" "DN + plain_remove_cxr" "${RESULTS_AUROC[dn_plain_remove_cxr]}"
printf "%-30s %10s\n" "DN + plain + PI" "${RESULTS_AUROC[dn_plain_pi]}"
printf "%-30s %10s\n" "DN+RR + plain" "${RESULTS_AUROC[dn_rr_plain]}"
printf "%-30s %10s\n" "DN+RR + plain_remove_cxr" "${RESULTS_AUROC[dn_rr_plain_remove_cxr]}"
printf "%-30s %10s\n" "DN+RR + plain + PI" "${RESULTS_AUROC[dn_rr_plain_pi]}"
echo ""
echo "Test AUPRC Results:"
echo "-------------------"
printf "%-30s %10s\n" "Configuration" "AUPRC"
echo "----------------------------------------"
printf "%-30s %10s\n" "DN + plain" "${RESULTS_AUPRC[dn_plain]}"
printf "%-30s %10s\n" "DN + plain_remove_cxr" "${RESULTS_AUPRC[dn_plain_remove_cxr]}"
printf "%-30s %10s\n" "DN + plain + PI" "${RESULTS_AUPRC[dn_plain_pi]}"
printf "%-30s %10s\n" "DN+RR + plain" "${RESULTS_AUPRC[dn_rr_plain]}"
printf "%-30s %10s\n" "DN+RR + plain_remove_cxr" "${RESULTS_AUPRC[dn_rr_plain_remove_cxr]}"
printf "%-30s %10s\n" "DN+RR + plain + PI" "${RESULTS_AUPRC[dn_rr_plain_pi]}"
echo ""
echo "========================================"
echo "All experiments completed!"
echo "========================================"
