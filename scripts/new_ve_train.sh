#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

for LR in 2e-5 1e-5; do
  python train_new_ve.py \
    --lr ${LR} \
    --output_path ../finetuned_model/vision_encoder_${LR} \

done   
