
CUDA_VISIBLE_DEVICES=4 python baseline_test.py --model_name_or_path StanfordAIMI/CheXagent-8b --output_path ./

CUDA_VISIBLE_DEVICES=5 python baseline_test.py --model_name_or_path mistralai/Pixtral-12B-2409 --output_path ./

CUDA_VISIBLE_DEVICES=6 python baseline_test.py --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct --output_path ./

CUDA_VISIBLE_DEVICES=7 python baseline_test.py --model_name_or_path microsoft/Phi-3.5-vision-instruct  --output_path ./