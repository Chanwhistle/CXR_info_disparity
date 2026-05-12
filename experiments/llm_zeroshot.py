"""
파인튜닝 없이 Llama 모델의 zero-shot 사망률 예측 스크립트.

dev/test set에 대해 모델을 그대로 실행하고 첫 번째 0/1 토큰을 예측으로 사용합니다.

실행 예시:
    HF_TOKEN=hf_xxx python llm_zeroshot.py \
        --use_discharge_note \
        --zeroshot \
        --base_img_dir ../saved_images \
        --base_rr_dir ../physionet.org/files/mimic-cxr/2.1.0/files \
        --output_path ../results/zeroshot
"""

import json
import logging
import os

import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

from utils import get_args, extract_first_binary, prepare_loader
from models.llama_model import load_model

# HF_TOKEN 환경변수로 private 모델 접근 (없으면 public 모델만 사용 가능)
_hf_token = os.environ.get("HF_TOKEN", None)
if _hf_token:
    from huggingface_hub import login
    login(_hf_token)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_prompt_end_index(token_ids, prompt_token_id):
    indices = (token_ids == prompt_token_id).nonzero(as_tuple=True)[0]
    return indices[-1].item() + 1 if len(indices) > 0 else 0


def run_inference_on_loader(model, processor, device, data_loader, args, set_type):
    """지정 split에 대한 zero-shot 예측 수행 후 결과 파일 저장."""
    os.makedirs(args.output_path, exist_ok=True)
    output_dict = {"data": []}
    predictions_file = os.path.join(args.output_path, f'{set_type}_predictions_{args.summary_type}.txt')

    with open(predictions_file, 'w') as f:
        for batch in tqdm(data_loader, total=len(data_loader)):
            batch_device = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                outputs = model(**batch_device)

            decoded = processor.decode(outputs, skip_special_tokens=True)
            output = {
                'label': int(batch_device['labels'][0]),
                'prediction': decoded,
                'binary': extract_first_binary(decoded)
            }
            output_dict['data'].append(output)
            f.write(json.dumps(output) + '\n')
        f.flush()

    preds, labels = zip(*[(ele['binary'], ele['label']) for ele in output_dict['data']])
    assert len(preds) == len(labels)

    with open(os.path.join(args.output_path, "score.txt"), "a") as f:
        f.write(f"{set_type} evaluation completed\n")
        f.write(f"Summary type        : {args.summary_type}\n")
        f.write(f"Use CXR image      : {args.use_cxr_image}\n")
        f.write(f"Use Radiology note : {args.use_rad_report}\n")
        f.write(f"Use Discharge note : {args.use_discharge_note}\n")
        f.write(f"Positive prediction : {sum(1 for p in preds if int(p)==1)}, Negative: {sum(1 for p in preds if int(p)==0)}\n")
        f.write(f"Positive f1         : {f1_score(preds, labels)}\n")
        f.write(classification_report(labels, preds, digits=4))
        f.write(f"Evaluated on {len(preds)} instances.\n\n")

    print(f"Inference complete. Predictions saved to {args.output_path}")
    return output_dict


def inference(args):
    model, processor = load_model(args, model_id=args.model_name_or_path, inference=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dev_loader = prepare_loader(args.summary_type, args, "dev", processor)
    test_loader = prepare_loader(args.summary_type, args, "test", processor)

    run_inference_on_loader(model, processor, device, dev_loader, args, "dev")
    run_inference_on_loader(model, processor, device, test_loader, args, "test")


if __name__ == "__main__":
    args = get_args()
    inference(args)
