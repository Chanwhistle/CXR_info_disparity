#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

# Example unique IDs - modify as needed
UNIQUE_IDS=(
  0x619017ba7206648b0ac3a81690589c
  0x5f23090cdf667ac1769cb333d75775
  0x2ffb6c576cadaa8c1ecc5444478037
  0xdca3b7ef61675bf996fbd8f23dd0a0
  0xc7a0a324c56130257ff7fa10bfceb7
  0x2ed5113a784fd1ef2bdf94ff98939f
  0x325d72634953a16a60c43aa3497ddc
  0x6259f9cbf8c4e2615d64a9e19ec259
  0x868a91f6152bf953646f13d2258841
  0x233142224a5179cfd99eb3831a3dc1
  0xeefb758dbcf82e8927d92095bf7ef2
  0xf0e76f89c93cd0ae0b24da85b0f899
  0x5f375e6084feabee35022184752049
  0xd0020fa1adafdc89e4294ceaec482d
  0x6a8242a0613ccb3fd412d94bfa5c9d
  0x353d82db826745c6752193b7eedad8
  0x6bf5aeb15b069730f317d251e61d72
  0x27535d2f3ea893e53cd53935d3b218
  0xbb141d1ece0116ec1add4b0ccf04ee
  0xfa9c2be6b9f98d3981ea63616bd6ef
)

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

for INDEX in "${UNIQUE_IDS[@]}"; do
  echo "=================================================="
  echo "Running 6-panel visualization for: $INDEX"
  echo "=================================================="

  python vit_patch_norm_heatmap.py \
    --unique_id "$INDEX" \
    --model_id meta-llama/Llama-3.2-11B-Vision-Instruct \
    --checkpoint_dir ../trained_models/dn+img \
    --out_dir ./vit_patch_norm_outputs \
    --metadata_path ../dataset/metadata.json \
    --metadata_image_path ../dataset/test_summarization/full-test-indent-images.json \
    --test_data_path ../dataset/test_summarization/total_output.jsonl \
    --base_img_dir ../saved_images \
    --base_rr_dir ../physionet.org/files/mimic-cxr/2.1.0/files \
    --do_occlusion \

done

echo "ALL JOBS FINISHED ✅"
