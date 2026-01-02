export TASK_NAME="out_hospital_mortality_30"
export LABEL_PATH=${TASK_NAME}/labels.json
export OUTPUT_PATH=${TASK_NAME} 

python create_data.py \
 --label_path ${LABEL_PATH} \
 --discharge_path ${NOTE_PATH} \
 --output_path ${OUTPUT_PATH} \
 --task_name ${TASK_NAME}