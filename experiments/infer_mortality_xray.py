"""
CXR 이미지 기반 학습된 모델 추론 스크립트.

train_mortality_xray.py로 학습된 모델(.pt)을 로드하여
JSON / .pt / .hdf5 형식의 평가 데이터에 대해 예측을 수행합니다.

실행 예시:
    python infer_mortality_xray.py \
        --eval_file ../out_hospital_mortality_30/dev_with_images.json \
        --cxr_root ../physionet.org/files/mimic-cxr-jpg/2.0.0 \
        --model_file ../trained_models/resnet_best.pt \
        --model_type resnet

출력 형식: <true_label>\t<prob_death>\t<pred_label>\t<hadm_id>\t<img_path>

Source: Modified from LCD Benchmark (https://github.com/Machine-Learning-for-Medical-Language/long-clinical-doc)
"""

import sys
import json
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import HfArgumentParser

from train_mortality_xray import dict_to_dataset, RESNET, MULTICHANNEL, BASELINE, CHEXNET, MIN_RES
from models.MultiChannelModel import MultiChannelMortalityPredictor, VariableLengthImageDataset, collate_fn


@dataclass
class InferenceArguments:
    eval_file: str = field(metadata={"help": ".json, .pt, or .hdf5 file for evaluation"})
    cxr_root: str = field(metadata={"help": "Path to mimic-cxr-jpg data root"})
    model_file: str = field(metadata={"help": "Pytorch model file (.pt) to load"})
    model_type: str = field(
        default='baseline',
        metadata={"choices": [RESNET, MULTICHANNEL, BASELINE, CHEXNET], "help": "Model architecture type"}
    )
    max_eval: int = field(default=-1, metadata={"help": "Max eval instances (-1 = all)"})


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = HfArgumentParser(InferenceArguments)
    infer_args, = parser.parse_args_into_dataclasses()

    model = torch.load(infer_args.model_file)

    if infer_args.model_type == MULTICHANNEL:
        eval_dataset = VariableLengthImageDataset(infer_args.eval_file)
        metadata = [
            {'hadm_id': eval_dataset.get_metadata(i), 'img_fn': eval_dataset.get_id(i)}
            for i in range(len(eval_dataset))
        ]
    else:
        with open(infer_args.eval_file, 'rt') as fp:
            dev_json = json.load(fp)
            eval_dataset, metadata = dict_to_dataset(dev_json, infer_args.cxr_root, max_size=infer_args.max_eval, train=False)

    model.eval()
    print("True label\tProb(death)\tPredicted label\tHADM_ID\tPath")

    with torch.no_grad():
        for ind in range(len(eval_dataset)):
            matrix, label = eval_dataset[ind]

            if infer_args.model_type == RESNET:
                padded_matrix = torch.zeros(1, 3, MIN_RES, MIN_RES)
                padded_matrix[0, 0] = matrix
                padded_matrix[0, 1] = matrix
                padded_matrix[0, 2] = matrix
            elif infer_args.model_type == MULTICHANNEL:
                # (num_images, 1, res, res) → (1, 1, num_images, res, res)
                padded_matrix = torch.permute(matrix, (1, 0, 2, 3)).unsqueeze(dim=0)
            else:
                padded_matrix = torch.zeros(1, MIN_RES, MIN_RES)
                padded_matrix[0] = matrix

            logits = model(padded_matrix.to(device)).cpu()
            probs = torch.softmax(logits, axis=1).numpy()[0]
            pred = np.argmax(logits.numpy(), axis=1)
            fn = metadata[ind]['img_fn']
            hadm_id = metadata[ind]['hadm_id']
            print("%d\t%f\t%d\t%s\t%s" % (label, probs[1], pred[0], hadm_id, fn))


if __name__ == '__main__':
    main(sys.argv[1:])
