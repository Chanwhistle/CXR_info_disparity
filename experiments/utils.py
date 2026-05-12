"""
학습/평가 공통 유틸리티: Trainer, 메트릭, argparse, 시드 설정.

AdapterOnlyTrainer: LoRA adapter + classifier만 선택적으로 저장하는 커스텀 HuggingFace Trainer.
  - save_model(): 모달리티에 따라 필요한 가중치만 저장
  - create_optimizer(): classifier에 별도 head_lr 적용 가능

주요 함수:
  - get_args()        : 학습/평가 공통 argparse 파싱
  - load_data()       : JSONL 파일 로드
  - prepare_loader()  : split(train/dev/test)에 맞는 DataLoader 생성
  - log_result()      : AUROC/AUPRC/ECE/Brier score 계산 및 파일 저장
  - compute_ece()     : Expected Calibration Error 계산
  - set_seed()        : 전체 랜덤 시드 고정
"""

import argparse
import json
import os
import random
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from transformers import Trainer, AutoProcessor

from dataloader import VLM_Dataset, custom_data_collator, CXRDecisionTree
from torch.utils.data import DataLoader
from PIL import Image


class AdapterOnlyTrainer(Trainer):
    """LoRA adapter + classifier head만 선택적으로 저장하는 Trainer."""

    def __init__(self, *args, **kwargs):
        self.use_cxr_image = kwargs.pop('use_cxr_image', False)
        self.head_lr = kwargs.pop('head_lr', None)
        super().__init__(*args, **kwargs)

    def save_model(self, output_dir: Optional[str] = "None", _internal_call: bool = False):
        """모달리티에 따라 필요한 모듈만 저장 (전체 모델 저장 불필요)."""
        os.makedirs(output_dir, exist_ok=True)
        args = self.model.args
        use_text = (
            getattr(args, 'use_discharge_note', False) or
            getattr(args, 'use_rad_report', False) or
            getattr(args, 'use_generated_rad_report', False)
        )
        use_image = getattr(args, 'use_cxr_image', False)
        print(f"\nSaving model to {output_dir}...")

        if use_image:
            # Vision LoRA adapter 저장
            try:
                vm_state = self.model.base_model.vision_model.get_adapter_state_dict()
                torch.save(vm_state, os.path.join(output_dir, "vm_adapter.bin"))
                print("Saved vision model LoRA adapter.")
            except Exception:
                torch.save(self.model.base_model.vision_model.state_dict(), os.path.join(output_dir, "vision_encoder.bin"))
                print("Warning: Saved full vision model (no adapter found).")

            torch.save(self.model.base_model.multi_modal_projector.state_dict(), os.path.join(output_dir, "multi_modal_projector.bin"))
            print("Saved multimodal projector.")

        if use_text:
            # LM LoRA adapter 저장 (텍스트 전용 또는 멀티모달)
            from models.llama_model import _get_lm
            lm = _get_lm(self.model.base_model)
            try:
                lm_state = lm.get_adapter_state_dict()
                torch.save(lm_state, os.path.join(output_dir, "lm_adapter.bin"))
                print("Saved language model LoRA adapter.")
            except Exception as e:
                print(f"Skipping LM adapter save: {e}")

        # Classifier는 항상 저장
        torch.save(self.model.classifier.state_dict(), os.path.join(output_dir, "classifier.bin"))
        print("Saved classification head.")

    def create_optimizer(self):
        """head_lr 지정 시 classifier와 LoRA/projector에 다른 lr 적용."""
        if self.head_lr is None:
            return super().create_optimizer()

        lora_projector_params = []
        classifier_params = []

        print("\n=== Trainable Parameters ===")
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'classifier' in name.lower():
                classifier_params.append(param)
            else:
                lora_projector_params.append(param)

        num_lora = sum(p.numel() for p in lora_projector_params)
        num_cls = sum(p.numel() for p in classifier_params)
        print(f"LoRA/Projector: {len(lora_projector_params)} tensors, {num_lora:,} elements")
        print(f"Classifier:     {len(classifier_params)} tensors, {num_cls:,} elements")

        optimizer_grouped_parameters = [
            {"params": lora_projector_params, "lr": self.args.learning_rate},
            {"params": classifier_params, "lr": self.head_lr},
        ]

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        if "lr" not in optimizer_kwargs:
            optimizer_kwargs["lr"] = self.args.learning_rate

        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if not hasattr(optimizer, 'defaults') or optimizer.defaults is None:
            optimizer.defaults = optimizer_kwargs

        self.optimizer = optimizer
        return optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        opt = optimizer if optimizer is not None else self.optimizer
        if opt is None:
            raise ValueError("Optimizer cannot be None when creating scheduler.")
        if not hasattr(opt, 'defaults') or opt.defaults is None:
            opt.defaults = {"lr": self.args.learning_rate}
        self.lr_scheduler = super().create_scheduler(num_training_steps, opt)
        return self.lr_scheduler


def extract_first_binary(s: str) -> int:
    """문자열에서 첫 번째 0 또는 1 추출 (zeroshot 예측 파싱용)."""
    for char in str(s):
        if char in ['0', '1']:
            return int(char)
    return 0


def prepare_loader(summary_type, args, set, processor):
    """split 이름에 따라 DataLoader 생성."""
    if set == "train":
        data_path, image_path = args.train_data_path, args.train_metadata_image_path
    elif set == "dev":
        data_path, image_path = args.dev_data_path, args.dev_metadata_image_path
    elif set == "test":
        data_path, image_path = args.test_data_path, args.test_metadata_image_path

    data = load_data(data_path, summary_type)
    if args.debug:
        data = data[:70]

    dataset = VLM_Dataset(
        args, data, image_path,
        args.use_cxr_image, args.use_rad_report,
        getattr(args, 'use_generated_rad_report', False),
        args.use_discharge_note, shuffle=False
    )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_data_collator(processor, use_cxr_image=args.use_cxr_image, summary_type=summary_type),
        pin_memory=True
    )


def load_data(path, summary_type):
    """JSONL 파일을 로드하고 summary_type 필드를 text로 사용."""
    dataset = []
    with open(path, "r") as f:
        for line in f:
            sample = json.loads(line)
            dataset.append({
                'id': sample['id'],
                'label': sample['label'],
                'text': sample[summary_type],
                'summary_type': summary_type
            })
    return dataset


def compute_metrics_auroc(eval_pred):
    """HuggingFace Trainer compute_metrics 콜백: AUROC + AUPRC 반환."""
    logits, labels = eval_pred
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    probabilities = F.softmax(torch.tensor(logits), dim=-1).detach().cpu().numpy()
    y_score = probabilities[:, 1]

    metrics = {}
    try:
        metrics["auroc"] = roc_auc_score(labels, y_score)
    except ValueError:
        metrics["auroc"] = float("nan")
    try:
        metrics["auprc"] = average_precision_score(labels, y_score)
    except ValueError:
        metrics["auprc"] = float("nan")

    return metrics


def load_adapter(current_state_dict, adapter_state):
    for key, value in adapter_state.items():
        if key in current_state_dict:
            current_state_dict[key].copy_(value)


def map_adapter_keys(adapter_state, adapter_name="language_model_adapter"):
    """저장된 LoRA 가중치 키를 현재 모델 state_dict 키로 매핑."""
    mapped_state = {}
    for key, value in adapter_state.items():
        parts = key.split('.')
        if 'lora_A' in parts or 'lora_B' in parts:
            weight_index = parts.index('weight')
            parts.insert(weight_index, adapter_name)
        mapped_state['.'.join(parts)] = value
    return mapped_state


def compute_ece(labels, pos_probs, n_bins=10):
    """Expected Calibration Error 계산 (10-bin)."""
    labels = np.asarray(labels).astype(int)
    pos_probs = np.asarray(pos_probs).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (pos_probs > lo) & (pos_probs <= hi)
        prop = np.mean(in_bin)
        if prop > 0:
            ece += np.abs(np.mean(pos_probs[in_bin]) - np.mean(labels[in_bin])) * prop

    return float(ece)


def _bootstrap_ci_auc_pr(labels, pos_probs, n_boot=2000, seed=42, stratified=True, alpha=0.05):
    """Bootstrap 방식으로 AUROC/AUPRC의 95% 신뢰 구간 추정."""
    labels = np.asarray(labels).astype(int)
    pos_probs = np.asarray(pos_probs).astype(float)
    rng = np.random.default_rng(seed)
    n = len(labels)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    roc_samples, pr_samples = [], []
    for _ in range(n_boot):
        if stratified:
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                break
            boot_idx = np.concatenate([
                rng.choice(pos_idx, size=len(pos_idx), replace=True),
                rng.choice(neg_idx, size=len(neg_idx), replace=True)
            ])
        else:
            boot_idx = rng.choice(np.arange(n), size=n, replace=True)

        yt = labels[boot_idx]
        ys = pos_probs[boot_idx]
        if len(np.unique(yt)) < 2:
            continue
        roc_samples.append(roc_auc_score(yt, ys))
        pr_samples.append(average_precision_score(yt, ys))

    roc_samples = np.asarray(roc_samples, dtype=float)
    pr_samples = np.asarray(pr_samples, dtype=float)

    if len(roc_samples) == 0:
        return None, None, 0

    lo, hi = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    auroc_ci = (float(np.percentile(roc_samples, lo)), float(np.percentile(roc_samples, hi)))
    auprc_ci = (float(np.percentile(pr_samples, lo)), float(np.percentile(pr_samples, hi)))
    return auroc_ci, auprc_ci, int(min(len(roc_samples), len(pr_samples)))


def log_result(args, labels, probs, output_path, set_type, compute_ci=True, n_boot=2000, seed=42):
    """AUROC/AUPRC/Brier/ECE 계산 후 score.txt에 append."""
    os.makedirs(output_path, exist_ok=True)
    labels = np.asarray(labels).astype(int)
    pos_probs = np.asarray([p[1] for p in probs], dtype=float)
    uniq = np.unique(labels)

    if len(uniq) < 2:
        auroc = auprc = brier = ece = auroc_ci = auprc_ci = "NA"
        n_boot_used = 0
        print(f"[WARN] Only one class in {set_type} labels: {uniq.tolist()}")
    else:
        auroc = float(roc_auc_score(labels, pos_probs))
        auprc = float(average_precision_score(labels, pos_probs))
        brier = float(brier_score_loss(labels, pos_probs))
        ece = compute_ece(labels, pos_probs)
        auroc_ci = auprc_ci = "NA"
        n_boot_used = 0
        if compute_ci:
            auroc_ci, auprc_ci, n_boot_used = _bootstrap_ci_auc_pr(labels, pos_probs, n_boot=n_boot, seed=seed)
            if auroc_ci is None:
                auroc_ci = auprc_ci = "NA"

    score_file = os.path.join(output_path, "score.txt")
    with open(score_file, "a") as f:
        f.write(f"{args.summary_type}{'_add_pi' if args.use_pi else ''} {set_type} evaluation completed\n")
        f.write(f"Num samples          : {len(labels)}\n")
        f.write(f"Num positives        : {int(labels.sum()) if isinstance(labels.sum(), (int, float)) else 'NA'}\n")
        f.write(f"AUROC               : {auroc}\n")
        f.write(f"AUPRC               : {auprc}\n")
        f.write(f"Brier Score         : {brier}\n")
        f.write(f"ECE (10 bins)       : {ece}\n")
        if compute_ci:
            f.write(f"Bootstrap n         : {n_boot_used}/{n_boot}\n")
            f.write(f"AUROC 95% CI        : {auroc_ci}\n")
            f.write(f"AUPRC 95% CI        : {auprc_ci}\n")
        f.write("\n")

    print(f"Inference complete. Predictions saved to {output_path}")


def set_seed(seed):
    """재현성을 위한 전체 랜덤 시드 고정."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


def get_args():
    """학습/추론 공통 argparse 설정. 경로는 모두 상대경로 기본값 제공."""
    parser = argparse.ArgumentParser(description="Llama mortality prediction: finetune / inference / zeroshot")

    # --- 모델 ---
    parser.add_argument("--model_name_or_path",
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="HuggingFace Hub 모델 ID 또는 로컬 경로")
    parser.add_argument("--checkpoint_dir",
        type=str, default="../trained_models",
        help="학습된 체크포인트 디렉토리")

    # --- 데이터 경로 (상대경로 기본값; 실행 위치에 맞게 수정 필요) ---
    parser.add_argument("--train_data_path",
        type=str, default="../dataset/train_summarization/total_output.jsonl")
    parser.add_argument("--dev_data_path",
        type=str, default="../dataset/dev_summarization/total_output.jsonl")
    parser.add_argument("--test_data_path",
        type=str, default="../dataset/test_summarization/total_output.jsonl")
    parser.add_argument("--metadata_path",
        type=str, default="../dataset/metadata.json")
    parser.add_argument("--train_metadata_image_path",
        type=str, default="../dataset/train_summarization/full-train-indent-images.json")
    parser.add_argument("--dev_metadata_image_path",
        type=str, default="../dataset/dev_summarization/full-dev-indent-images.json")
    parser.add_argument("--test_metadata_image_path",
        type=str, default="../dataset/test_summarization/full-test-indent-images.json")

    # --- 외부 데이터 루트 경로 (반드시 지정) ---
    parser.add_argument("--base_img_dir",
        required=True, help="저장된 CXR 이미지 폴더 (train/dev/test 하위 폴더 포함)")
    parser.add_argument("--base_rr_dir",
        required=True, help="MIMIC-CXR 방사선 보고서 폴더 (physionet.org/.../files)")

    # --- 출력 ---
    parser.add_argument("--output_path",
        type=str, default="./trained_models", help="모델/결과 저장 경로")
    parser.add_argument("--data_dir",
        type=str, default="../dataset",
        help="dataset 루트 디렉토리 (summarize_dn.py에서 입력 JSONL 탐색 기준)")

    # --- 학습 하이퍼파라미터 ---
    parser.add_argument("--summary_type",  type=str, default='plain')
    parser.add_argument("--set_name",      type=str, default='train')
    parser.add_argument("--num_epochs",    type=int, default=3)
    parser.add_argument("--lr",            type=float, default=5e-5)
    parser.add_argument("--head_lr",       type=float, default=None,
        help="classifier 전용 lr (None이면 --lr 사용)")
    parser.add_argument("--batch_size",    type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--seed",          type=int, default=42)

    # --- 모달리티 플래그 ---
    parser.add_argument("--use_cxr_image",             action="store_true")
    parser.add_argument("--use_rad_report",             action="store_true")
    parser.add_argument("--use_generated_rad_report",   action="store_true")
    parser.add_argument("--use_discharge_note",         action="store_true")
    parser.add_argument("--use_pi",                     action="store_true",
        help="나이/인종 등 개인정보 포함 여부")

    # --- 실행 모드 플래그 ---
    parser.add_argument("--debug",      action="store_true")
    parser.add_argument("--finetune",   action="store_true")
    parser.add_argument("--summarize",  action="store_true")
    parser.add_argument("--zeroshot",   action="store_true")
    parser.add_argument("--wandb",      action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")

    return parser.parse_args()
