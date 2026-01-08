
RR_DIR="../physionet.org/files/mimic-cxr/2.1.0/files"

for INDEX in 56 71 143 179 221 1108 1149 1150 1219 1322; do
    CUDA_VISIBLE_DEVICES=1 python visualize_attention_map.py \
    --checkpoint_path ../trained_models/dn+img/checkpoint \
    --use_discharge_note \
    --index ${INDEX} \
    --base_img_dir ../saved_images \
    --target_layer all
done   
