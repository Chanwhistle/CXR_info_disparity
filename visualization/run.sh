#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# Example unique IDs - modify as needed
UNIQUE_IDS=(
  # 0x619017ba7206648b0ac3a81690589c
  # 0x5f23090cdf667ac1769cb333d75775
  # 0x2ffb6c576cadaa8c1ecc5444478037
  # 0xdca3b7ef61675bf996fbd8f23dd0a0
  # 0xc7a0a324c56130257ff7fa10bfceb7
  # 0x2ed5113a784fd1ef2bdf94ff98939f
  # 0x325d72634953a16a60c43aa3497ddc
  # 0x6259f9cbf8c4e2615d64a9e19ec259
  # 0x868a91f6152bf953646f13d2258841
  # 0x233142224a5179cfd99eb3831a3dc1
  # 0xeefb758dbcf82e8927d92095bf7ef2
  # 0xf0e76f89c93cd0ae0b24da85b0f899
  # 0x5f375e6084feabee35022184752049
  # 0xd0020fa1adafdc89e4294ceaec482d
  # 0x6a8242a0613ccb3fd412d94bfa5c9d
  # 0x353d82db826745c6752193b7eedad8
  # 0x6bf5aeb15b069730f317d251e61d72
  # 0x27535d2f3ea893e53cd53935d3b218
  # 0xbb141d1ece0116ec1add4b0ccf04ee
  # 0xfa9c2be6b9f98d3981ea63616bd6ef
  
  # Positive cases
  # 0x27a25b57c7e7a7582094ed67bf470d
  # 0xcb4bcad3e8b61c0998d07d01bd8c8f
  # 0xf2977385a4be8500737657330572c6
  # 0x6259f9cbf8c4e2615d64a9e19ec259
  # 0x15fa03917399801b55c1bcd0704bcd
  # 0xeefb758dbcf82e8927d92095bf7ef2
  # 0xe0425507b65d835e92cc85442f188a
  # 0x54e6576c031f1f1863ba8262aa3d48
  # 0x6d0d1d42289188a32381c5f063d719
  # 0x833df61b50ac8c0d093580f8c0083e
  # 0xe6d6a0f0185795c2461e99a9c964a2
  # 0x216c23dc9020954df8742dd4aab23a
  # 0x2a481cc21ba20e6867817d3b662d9b
  # 0x43fb4f59f2d48a724b132c084fb3a5
  # 0x8b72946eb927665047837d8ab426ff

  0x8375f0076fb9b736c7f6bf6a275a78
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
    --out_dir ./vit_patch_norm_outputs_example \
    --metadata_path ../dataset/metadata.json \
    --metadata_image_path ../dataset/train_summarization/full-train-indent-images.json \
    --test_data_path ../dataset/train_summarization/total_output.jsonl \
    --base_img_dir ../saved_images \
    --base_rr_dir ../physionet.org/files/mimic-cxr/2.1.0/files \
    --load_in_4bit \
    --do_occlusion \

done

echo "ALL JOBS FINISHED ✅"
