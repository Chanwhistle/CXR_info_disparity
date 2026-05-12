"""
Llama 모델을 사용한 퇴원 노트 요약 스크립트.

입력 JSONL의 original_note 필드를 지정된 summary_type 형식(plain/risk_factor/timeline 등)으로
요약하여 새 JSONL로 저장합니다.

실행 예시:
    HF_TOKEN=hf_xxx python summarize_dn.py \
        --set_name train \
        --summary_type plain \
        --data_dir ../dataset \
        --output_path ../dataset/train_summarization \
        --base_img_dir ../saved_images \
        --base_rr_dir ../physionet.org/files/mimic-cxr/2.1.0/files \
        --summarize

Note: HF_TOKEN 환경변수로 private 모델 접근 가능.
"""

import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import get_args, load_data
from models.llama_model import load_model
from dataloader import VLM_Dataset, custom_data_collator

# HF_TOKEN 환경변수로 private 모델 접근 (없으면 public 모델만 사용 가능)
_hf_token = os.environ.get("HF_TOKEN", None)
if _hf_token:
    from huggingface_hub import login
    login(_hf_token)


def summarize(args):
    model, processor = load_model(args, model_id=args.model_name_or_path, inference=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model.config.temperature = 0.1

    # 입력 JSONL 경로: --data_dir / {set_name}_summarization / {set_name}.jsonl
    input_path = os.path.join(args.data_dir, f"{args.set_name}_summarization", f"{args.set_name}.jsonl")
    data = load_data(input_path, "original_note")

    if args.set_name == "train":
        data_image_path = args.train_metadata_image_path
    elif args.set_name == "dev":
        data_image_path = args.dev_metadata_image_path
    else:
        data_image_path = args.test_metadata_image_path

    dataset = VLM_Dataset(
        args, data, data_image_path,
        args.use_cxr_image, args.use_rad_report,
        use_discharge_note=False,
        shuffle=False, summarize=True
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_data_collator(processor, use_cxr_image=args.use_cxr_image, summary_type="original_note"),
    )

    os.makedirs(args.output_path, exist_ok=True)
    summary_file = os.path.join(args.output_path, f'{args.summary_type}_output.jsonl')

    with open(summary_file, 'w') as f:
        for idx, sample in enumerate(tqdm(data_loader, total=len(data_loader))):
            sample_device = {k: v.to(device) for k, v in sample.items() if isinstance(v, torch.Tensor)}
            with torch.no_grad():
                outputs = model(**sample_device)

            decoded = processor.decode(outputs, skip_special_tokens=True)
            output = {
                'id': sample['ids'][0],
                'label': int(sample_device['labels'][0]),
                'original_note': [dataset[idx]['original_note'], len(processor.tokenizer.encode(dataset[idx]['original_note']))],
                f'{args.summary_type}': [decoded[9:], len(outputs[2:])],
            }
            f.write(json.dumps(output) + '\n')
            f.flush()


if __name__ == "__main__":
    args = get_args()
    # data_dir 인자가 없으면 기본 상대경로 사용
    if not hasattr(args, 'data_dir') or args.data_dir is None:
        args.data_dir = "../dataset"
    summarize(args)
