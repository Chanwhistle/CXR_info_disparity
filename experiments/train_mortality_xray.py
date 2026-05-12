"""
CXR 이미지 기반 사망률 예측 모델 학습 스크립트.

지원 모델: ResNet34, ViT-L/16, MultiChannel(ViT+Transformer), MultiModal(CNN+ViT), Baseline CNN
입력 포맷: JSON (add_images_to_json.py 출력) 또는 캐시된 .pt / .hdf5 파일

실행 예시 (ResNet):
    python train_mortality_xray.py \
        --train_file ../out_hospital_mortality_30/train_with_images.json \
        --eval_file ../out_hospital_mortality_30/dev_with_images.json \
        --cxr_root ../physionet.org/files/mimic-cxr-jpg/2.0.0 \
        --model resnet \
        --model_filename ../trained_models/resnet_best.pt

Adapted from pytorch-cxr (https://github.com/jinserk/pytorch-cxr)
Source: Modified from LCD Benchmark (https://github.com/Machine-Learning-for-Medical-Language/long-clinical-doc)
"""

import sys
import json
from os.path import join, exists, dirname
from dataclasses import dataclass, field

from tqdm import tqdm

from models.BaselineModel import BaselineMortalityPredictor
from models.MultiChannelModel import MultiChannelMortalityPredictor, VariableLengthImageDataset, collate_fn
from models.MultiModalModel import MultiModalMortalityPredictor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, Subset
import torch.nn.functional as F
import torchvision.transforms as tfms
import torchvision.models as models
import torchxrayvision as xrv
from transformers import HfArgumentParser
from PIL import Image
import imageio
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, precision_recall_curve
from scipy.special import softmax

RESNET = 'resnet'
BASELINE = "baseline"
MULTICHANNEL = "mc"
CHEXNET = "chexnet"
VIT = "vit"
MULTIMODAL = "multimodal"

# Adapted from pytorch-cxr
MIN_RES = 512
MEAN = 0.0
STDEV = 0.229

cxr_train_transforms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.RandomAffine((-5, 5), translate=None, scale=None, shear=(0.9, 1.1)),
    tfms.RandomResizedCrop((MIN_RES, MIN_RES), scale=(0.5, 0.75), ratio=(0.95, 1.05), interpolation=Image.Resampling.LANCZOS),
    tfms.ToTensor(),
    tfms.Normalize((MEAN,), (STDEV,))
])

cxr_infer_transforms = tfms.Compose([
    tfms.ToPILImage(),
    tfms.Resize((MIN_RES, MIN_RES), interpolation=Image.Resampling.LANCZOS),
    tfms.CenterCrop(MIN_RES),
    tfms.ToTensor(),
    tfms.Normalize((MEAN,), (STDEV,))
])


def dict_to_dataset(mimic_dict, mimic_path, max_size=-1, train=True, image_selection='last'):
    """JSON 데이터를 TensorDataset으로 변환. image_selection='last'이면 입원일 이후 가장 최근 portable AP 선택."""
    num_insts = 0
    for inst in mimic_dict['data']:
        if 'images' in inst.keys() and len(inst['images']) > 0:
            num_insts += 1

    metadata = []
    print("Found %d instances in this data" % (num_insts))
    if max_size >= 0 and num_insts > max_size:
        num_insts = max_size
        print("Using %d instances due to passed in argument" % (num_insts))

    if image_selection == 'last':
        insts = []
        labels = []
        for inst in tqdm(mimic_dict['data']):
            inst_metadata = {}
            if 'images' not in inst or len(inst['images']) == 0:
                continue
            adm_dt = inst["debug_features"]["ADMITTIME"].split(" ")[0].replace("-","")
            sorted_images = sorted(inst['images'], key=lambda x: x['StudyDate'])
            for image in sorted_images:
                if str(image["StudyDate"]) >= adm_dt and image["PerformedProcedureStepDescription"] == "CHEST (PORTABLE AP)":
                    img_path = join(mimic_path, 'files', image['path'][:-4] + '_%d_resized.jpg' % (MIN_RES))
                    if not exists(img_path):
                        raise Exception("Path to image does not exist: %s" % (img_path))
                    img_data = imageio.imread(img_path, mode='F')
                    insts.append(cxr_train_transforms(img_data) if train else cxr_infer_transforms(img_data))
                    labels.append(inst['out_hospital_mortality_30'])
                    inst_metadata['img_fn'] = img_path
                    inst_metadata['hadm_id'] = inst["debug_features"]["HADM_ID"]
                    metadata.append(inst_metadata)
                    break

            if len(labels) >= num_insts:
                break

        padded_matrix = torch.stack(insts).squeeze()
        dataset = TensorDataset(padded_matrix, torch.tensor(labels))
    else:
        raise NotImplementedError("Image selection method %s is not implemented" % (image_selection))

    return dataset, metadata


def run_one_eval(model, eval_dataset, device, model_type):
    """모델 평가: 정확도/F1/AUROC/PRC 계산."""
    num_correct = num_wrong = 0
    model.eval()

    preds = np.zeros(len(eval_dataset))
    test_labels = np.zeros(len(eval_dataset))
    test_probs = []

    with torch.no_grad():
        dev_loss = 0
        for ind in range(0, len(eval_dataset)):
            if len(eval_dataset[ind]) == 3:
                matrix, text, label = eval_dataset[ind]
            else:
                matrix, label = eval_dataset[ind]

            test_labels[ind] = label

            if model_type == RESNET or model_type == VIT:
                padded_matrix = torch.zeros(1, 3, MIN_RES, MIN_RES)
                padded_matrix[0, 0] = matrix
                padded_matrix[0, 1] = matrix
                padded_matrix[0, 2] = matrix
                logits = model(padded_matrix.to(device))
            elif model_type == MULTICHANNEL or model_type == MULTIMODAL:
                padded_matrix = torch.permute(matrix, (1, 0, 2, 3)).unsqueeze(dim=0)
                if model_type == MULTICHANNEL:
                    logits = model(padded_matrix.to(device))
                elif model_type == MULTIMODAL:
                    logits = model(img_matrix=padded_matrix.to(device), text=[text])
            else:
                padded_matrix = matrix.unsqueeze(dim=0)
                logits = model(padded_matrix.to(device))

            label = torch.tensor(label)
            loss = F.cross_entropy(logits, label.unsqueeze(dim=0).to(device))
            dev_loss = loss.item()
            pred = np.argmax(logits.cpu().numpy(), axis=1)
            preds[ind] = pred
            test_probs.append(torch.softmax(logits.cpu(), dim=1)[:, 1].numpy())

            if pred == label:
                num_correct += 1
            else:
                num_wrong += 1

    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average=None)
    rec = recall_score(test_labels, preds, average=None)
    prec = precision_score(test_labels, preds, average=None)
    prev = test_labels.sum() / len(test_labels)
    auroc = roc_auc_score(y_true=test_labels, y_score=test_probs)
    precs, recs, thresholds = precision_recall_curve(y_true=test_labels, y_score=test_probs, drop_intermediate=True)

    # threshold 기반 최대 F1 탐색
    max_f1, max_threshold = -1, 0.0
    max_f1_prec = max_f1_rec = 0
    for dp_ind, threshold in enumerate(thresholds):
        dp_prec, dp_rec = precs[dp_ind], recs[dp_ind]
        dp_f1 = 2 * dp_prec * dp_rec / (dp_prec + dp_rec)
        if dp_f1 > max_f1:
            max_f1 = dp_f1
            max_threshold = threshold
            max_f1_rec = dp_rec
            max_f1_prec = dp_prec

    return {
        'dev_loss': dev_loss, 'acc': acc, 'f1': f1, 'rec': rec, 'prec': prec,
        'prevalence': prev, 'auroc': auroc,
        'prc': {'prec': max_f1_prec, 'rec': max_f1_rec, 'f1': max_f1, 'thresh': max_threshold}
    }


@dataclass
class TrainingArguments:
    train_file: str = field(metadata={"help": ".json, .pt, or .hdf5 file for training"})
    eval_file: str = field(metadata={"help": ".json, .pt, or .hdf5 file for evaluating during training"})
    cxr_root: str = field(default=None, metadata={"help": "Path to mimic-cxr-jpg data root"})
    num_training_epochs: int = field(default=10, metadata={"help": "Number of training epochs"})
    model_filename: str = field(default=None, metadata={"help": "Filename to write saved .pt model"})
    seed: int = field(default=42, metadata={"help": "Random seed"})
    batch_size: int = field(default=10, metadata={"help": "Training batch size"})
    eval_freq: int = field(default=10, metadata={"help": "Evaluation frequency (epochs)"})
    learning_rate: float = field(default=0.001, metadata={"help": "AdamW learning rate"})
    model: str = field(
        default='baseline',
        metadata={"choices": [RESNET, MULTICHANNEL, BASELINE, CHEXNET, VIT, MULTIMODAL], "help": "Model type"}
    )
    label_field: str = field(default="out_hospital_mortality_30", metadata={"help": "Label field in JSON"})
    max_train: int = field(default=-1, metadata={"help": "Max training instances (-1 = all)"})
    max_eval: int = field(default=-1, metadata={"help": "Max evaluation instances (-1 = all)"})
    img_model: str = field(default=None, metadata={"help": "Pretrained image model path (multimodal only)"})
    txt_model: str = field(default=None, metadata={"help": "Pretrained text model path (multimodal only)"})
    ga: int = field(default=1, metadata={"help": "Gradient accumulation steps"})


def main(args):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        sys.stderr.write("CUDA not available. Exiting (CPU training is too slow for this task).\n")
        sys.exit(-1)

    parser = HfArgumentParser(TrainingArguments)
    training_args, = parser.parse_args_into_dataclasses()
    batch_size = training_args.batch_size
    eval_freq = training_args.eval_freq
    ga = training_args.ga

    print("Training args: %s" % str(training_args))
    torch.cuda.manual_seed(training_args.seed)
    learning_rate = training_args.learning_rate

    # 모델 초기화
    if training_args.model == RESNET:
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(512, 2)
    elif training_args.model == CHEXNET:
        raise NotImplementedError("Chexnet is not working now.")
    elif training_args.model == MULTICHANNEL:
        model = MultiChannelMortalityPredictor((MIN_RES, MIN_RES))
    elif training_args.model == VIT:
        model = models.vision_transformer.vit_l_16(models.vision_transformer.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        model.heads.head = nn.Linear(1024, 2)
    elif training_args.model == MULTIMODAL:
        model = MultiModalMortalityPredictor(img_model=training_args.img_model, text_model=training_args.txt_model)
    else:
        model = BaselineMortalityPredictor((MIN_RES, MIN_RES))

    model.zero_grad()
    model = model.to(device)
    loss_fct = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("Loading training data...")
    if training_args.train_file.endswith('.json'):
        with open(training_args.train_file, 'rt') as fp:
            train_json = json.load(fp)
            if training_args.model == MULTICHANNEL:
                raise Exception("For multi-channel models, dataset must be pre-processed first with preprocess_mc_instances.py!")
            else:
                train_dataset, _ = dict_to_dataset(train_json, training_args.cxr_root, max_size=training_args.max_train, train=True)
            out_file = join(dirname(training_args.train_file), 'train_cache_res=%d.pt' % MIN_RES)
            torch.save(train_dataset, out_file)
    elif training_args.train_file.endswith('.pt'):
        train_dataset = torch.load(training_args.train_file)
        if training_args.max_train > 0 and training_args.max_train < len(train_dataset):
            train_dataset = Subset(train_dataset, list(range(training_args.max_train)))
    elif training_args.train_file.endswith('.hdf5'):
        train_dataset = VariableLengthImageDataset(training_args.train_file)
        if training_args.max_train > 0:
            train_dataset = Subset(train_dataset, list(range(training_args.max_train)))

    print("Loading evaluation data...")
    if training_args.eval_file.endswith('.json'):
        with open(training_args.eval_file, 'rt') as fp:
            dev_json = json.load(fp)
            if training_args.model == MULTICHANNEL:
                raise Exception("For multi-channel models, dataset must be pre-processed first!")
            else:
                dev_dataset, metadata = dict_to_dataset(dev_json, training_args.cxr_root, max_size=training_args.max_eval, train=False)
            out_file = join(dirname(training_args.eval_file), 'dev_cache_res=%d.pt' % MIN_RES)
            torch.save(dev_dataset, out_file)
    elif training_args.eval_file.endswith('.pt'):
        dev_dataset = torch.load(training_args.eval_file)
    elif training_args.eval_file.endswith('.hdf5'):
        dev_dataset = VariableLengthImageDataset(training_args.eval_file)

    sampler = RandomSampler(train_dataset)
    if training_args.model in [MULTICHANNEL, MULTIMODAL]:
        dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size, collate_fn=collate_fn, num_workers=8)
    else:
        dataloader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size)

    best_loss = None
    all_dev_results = {}

    for epoch in range(training_args.num_training_epochs):
        model.train()
        epoch_loss = 0.0
        accum_steps = 0

        for batch in tqdm(dataloader, desc="Epoch %d" % epoch):
            if len(batch) == 3:
                batch_images, batch_text, batch_labels = batch
            elif len(batch) == 2:
                batch_images, batch_labels = batch

            batch_images = torch.stack(tuple(t.to(device) for t in batch_images))
            batch_labels = torch.stack(tuple(t.to(device) for t in batch_labels))

            if training_args.model == RESNET:
                rgb_batch = torch.repeat_interleave(batch_images.unsqueeze(1), 3, dim=1)
                logits = model(rgb_batch)
            elif training_args.model == VIT:
                rgb_batch = torch.repeat_interleave(batch_images.unsqueeze(1), 3, dim=1)
                logits = model(rgb_batch)
            elif training_args.model == MULTICHANNEL:
                logits = model(batch_images.unsqueeze(1))
            elif training_args.model == MULTIMODAL:
                logits = model(img_matrix=batch_images.unsqueeze(1), text=batch_text)
            else:
                logits = model(batch_images.unsqueeze(1))

            loss = loss_fct(logits, batch_labels)
            loss.backward()
            accum_steps += 1

            if accum_steps % ga == 0:
                opt.step()
                opt.zero_grad()
                epoch_loss += loss.item()
                accum_steps = 0

        print("Epoch %d loss: %0.9f" % (epoch, epoch_loss))

        if epoch % eval_freq == 0:
            dev_results = run_one_eval(model, dev_dataset, device, training_args.model)
            thing_to_minimize = -dev_results['auroc']  # AUROC 최대화
            for k, v in dev_results.items():
                print("Dev %s = %s" % (k, str(v)))
                all_dev_results.setdefault(k, []).append(v)

            if not best_loss or thing_to_minimize < best_loss:
                best_loss = thing_to_minimize
                if training_args.model_filename:
                    print("Saving model")
                    torch.save(model, training_args.model_filename)

    final_dev_results = run_one_eval(model, dev_dataset, device, training_args.model)
    print("Final dev results: %s" % str(final_dev_results))
    print("All dev results: %s" % str(all_dev_results))


if __name__ == '__main__':
    main(sys.argv[1:])
