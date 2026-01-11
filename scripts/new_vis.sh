#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

UNIQUE_IDS=(
#   0x15fa03917399801b55c1bcd0704bcd
  0x5f23090cdf667ac1769cb333d75775
#   0x717b5f78ea1a92dbb7e3a3c2adda3f
#   0x54e6576c031f1f1863ba8262aa3d48
#   0x216c23dc9020954df8742dd4aab23a
)

for INDEX in "${UNIQUE_IDS[@]}"; do
  echo "=================================================="
  echo "Running 3-stage explainability for: $INDEX"
  echo "=================================================="

  python visualization_new.py \
    --unique_id "$INDEX" \
    --checkpoint_dir ../trained_models/dn+img \
    --out_dir ./attention_outputs \
    --do_occlusion \
    --occ_micro_batch 8 \
    --occ_stride 2 \
    --cross_layers 1 \
    --vit_layers 1
done

echo "ALL JOBS FINISHED ✅"
