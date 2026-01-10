#!/bin/bash

# Google Drive 업로드를 위한 압축 스크립트
# 불필요한 파일을 제외하고 압축합니다.

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== 압축 파일 생성 스크립트 ===${NC}"
echo ""

OUTPUT_FILE="CXR_info_disparity_$(date +%Y%m%d_%H%M%S).tar.gz"

echo -e "${YELLOW}압축 파일 생성 중: $OUTPUT_FILE${NC}"
echo "불필요한 파일들을 제외하고 압축합니다..."
echo ""

tar -czf "$OUTPUT_FILE" \
  --exclude='dataset' \
  --exclude='physionet.org' \
  --exclude='saved_images' \
  --exclude='trained_models' \
  --exclude='cache_images' \
  --exclude='attention_outputs' \
  --exclude='inference_result' \
  --exclude='wandb' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*.pyd' \
  --exclude='.Python' \
  --exclude='env' \
  --exclude='venv' \
  --exclude='ENV' \
  --exclude='*.egg-info' \
  --exclude='.pytest_cache' \
  --exclude='.ipynb_checkpoints' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.bin' \
  --exclude='*.ckpt' \
  --exclude='*.log' \
  --exclude='.git' \
  --exclude='.vscode' \
  --exclude='.idea' \
  --exclude='.DS_Store' \
  --exclude='*.swp' \
  --exclude='*.swo' \
  --exclude='*~' \
  --exclude='former' \
  --exclude='analyze' \
  --exclude="$OUTPUT_FILE" \
  .

echo ""
echo -e "${GREEN}압축 완료!${NC}"
echo ""
echo "압축 파일: $OUTPUT_FILE"
echo "크기: $(du -h "$OUTPUT_FILE" | cut -f1)"
echo ""
echo "다음 단계:"
echo "  1. Google Drive 웹사이트로 이동"
echo "  2. 파일 업로드"
echo "  3. 또는 rclone으로 업로드: rclone copy $OUTPUT_FILE bch:CXR_info_disparity/"
echo ""
