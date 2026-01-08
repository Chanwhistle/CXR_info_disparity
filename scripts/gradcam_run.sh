for INDEX in "0xf2977385a4be8500737657330572c6" "0x5558d145baf95199098c5ab393c8c1" "0x5f23090cdf667ac1769cb333d75775" "0x316e3d526bd3eab2889b31efef54e2"; do
  CUDA_VISIBLE_DEVICES=5 python visualize_attention_map.py \
    --checkpoint_path ../trained_models/dn+img/checkpoint \
    --use_discharge_note \
    --unique_id "${INDEX}" \
    --base_img_dir ../saved_images \
    --target_layer all
done
