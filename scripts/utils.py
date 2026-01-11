# 표준 라이브러리
import argparse
import json
import copy
import os

# 서드파티 라이브러리
from typing import Optional
import torch
import torch.nn.functional as F
from sklearn.metrics import *
from transformers import Trainer
from dataloader import *
import numpy as np


from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

class AdapterOnlyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # kwargs에서 꺼내되, 없으면 False (안전장치)
        self.use_cxr_image = kwargs.pop('use_cxr_image', False)
        self.head_lr = kwargs.pop('head_lr', None)  # head_lr 추가
        super().__init__(*args, **kwargs)

    def save_model(self, output_dir: Optional[str] = "None", _internal_call: bool = False):
        """
        학습된 모듈만 선별적으로 저장합니다.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 모델의 args에 접근하여 모달리티 사용 여부 확인
        args = self.model.args
        
        # 1. Text 사용 여부 확인 (앞서 정의한 로직)
        use_text = (
            getattr(args, 'use_discharge_note', False) or 
            getattr(args, 'use_rad_report', False) or 
            getattr(args, 'use_generated_rad_report', False)
        )
        
        # 2. Image 사용 여부 확인 (init에서 받거나 model args에서 확인)
        use_image = getattr(args, 'use_cxr_image', False)

        print(f"\nSaving model to {output_dir}...")

        # --- [Image Part] Vision Model & Projector ---
        if use_image:
            # 1) Vision Model (LoRA or Full)
            try:
                # LoRA가 적용된 경우
                vm_state = self.model.base_model.vision_model.get_adapter_state_dict()
                torch.save(vm_state, os.path.join(output_dir, "vm_adapter.bin"))
                print("Saved vision model LoRA adapter.")
            except:
                # LoRA가 없거나 Full finetuning인 경우 (혹은 get_adapter_state_dict 실패 시)
                # 주의: Vision Encoder 전체를 저장하면 용량이 큽니다. 필요시 state_dict 필터링 필요.
                torch.save(self.model.base_model.vision_model.state_dict(), os.path.join(output_dir, "vision_encoder.bin"))
                print("Warning: Saved full vision model (No adapter found).")

            # 2) Projector (보통 Full Fine-tuning 하므로 state_dict 저장)
            torch.save(self.model.base_model.multi_modal_projector.state_dict(), os.path.join(output_dir, "multi_modal_projector.bin"))
            print("Saved multimodal projector.")

        # --- [Text Part] Language Model ---
        if use_text:
            # Text Only 혹은 Multimodal일 때만 LM Adapter 저장
            # Image Only일 때는 LM이 Freeze 되어 있으므로 저장할 필요 없음
            try:
                lm_state = self.model.base_model.language_model.get_adapter_state_dict()
                torch.save(lm_state, os.path.join(output_dir, "lm_adapter.bin"))
                print("Saved language model LoRA adapter.")
            except Exception as e:
                print(f"Skipping LM adapter save (maybe not trained?): {e}")

        # --- [Classifier] ---
        # Classifier는 항상 학습하므로 무조건 저장
        torch.save(self.model.classifier.state_dict(), os.path.join(output_dir, "classifier.bin"))
        print("Saved classification head.")
    
    def create_optimizer(self):
        """
        LoRA와 Projector는 기본 lr로, Classifier만 head_lr로 학습
        head_lr이 None이면 기본 optimizer 사용
        """
        if self.head_lr is None:
            # head_lr이 없으면 기본 optimizer 사용
            return super().create_optimizer()
        
        # Parameter groups 분리
        lora_projector_params = []  # LoRA + Projector (기본 lr)
        classifier_params = []      # Classifier만 (head_lr)
        
        print("\n=== Trainable Parameters ===")
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'classifier' in name.lower():
                classifier_params.append(param)
            else:
                lora_projector_params.append(param)
        
        # 각 그룹별 실제 파라미터 숫자(숫자 개수) 계산
        num_lora_params = sum(p.numel() for p in lora_projector_params)
        num_classifier_params = sum(p.numel() for p in classifier_params)

        print(f"Total LoRA/Projector: {len(lora_projector_params)} tensors, {num_lora_params:,} elements")
        print(f"Total Classifier: {len(classifier_params)} tensors, {num_classifier_params:,} elements")
        
        # Optimizer 설정
        optimizer_grouped_parameters = [
            {
                "params": lora_projector_params,
                "lr": self.args.learning_rate,  # 기본 LR (LoRA + Projector용)
            },
            {
                "params": classifier_params,
                "lr": self.head_lr,  # Head LR (Classifier만)
            }
        ]
        
        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        
        # defaults를 먼저 설정 - 모든 필요한 키를 포함
        # 1. optimizer_kwargs에 lr이 반드시 포함되도록 함
        if "lr" not in optimizer_kwargs:
            optimizer_kwargs["lr"] = self.args.learning_rate
        
        # 2. Optimizer 생성
        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        
        # [중요] 3. 일부 Optimizer에서 defaults가 None인 경우를 대비해 직접 딕셔너리 할당
        if not hasattr(optimizer, 'defaults') or optimizer.defaults is None:
            optimizer.defaults = optimizer_kwargs

        # [중요] 4. Trainer의 멤버 변수에 직접 할당 (Traceback 방지)
        self.optimizer = optimizer
        
        return optimizer
    
    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """
        optimizer가 None으로 들어올 경우 self.optimizer를 사용하도록 강제
        """
        # 인자로 받은 optimizer가 없으면 self.optimizer 사용
        opt = optimizer if optimizer is not None else self.optimizer
        
        if opt is None:
            # 이 시점에도 None이면 optimizer 생성이 실패한 것임
            raise ValueError("Optimizer cannot be None when creating scheduler.")

        # scheduler 생성 시점에 한 번 더 체크
        if not hasattr(opt, 'defaults') or opt.defaults is None:
            opt.defaults = {"lr": self.args.learning_rate}
        
        # super()를 호출할 때 확실히 채워진 optimizer를 전달
        self.lr_scheduler = super().create_scheduler(num_training_steps, opt)
        return self.lr_scheduler

    # def load_model(self, checkpoint_dir: str):
    #     if self.use_cxr_image:
    #         try:
    #             self.model.base_model.vision_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "vision_encoder.bin"), map_location="cpu", weights_only=True))
    #             print("Loaded vision model.")
    #         except:
    #             print("No vision model!")
            
    #         try:
    #             self.model.base_model.vision_model.load_adapter(peft_model_id=f"{checkpoint_dir}/vm_adapter")
    #             print("Loaded vision model LoRA adapter.")
    #         except:
    #             print("No vision model LoRA adapter!")

    #             self.model.base_model.multi_modal_projector.load_state_dict(torch.load(os.path.join(checkpoint_dir, "multi_modal_projector.bin"), map_location="cpu", weights_only=True))
    #             print("Loaded multimodal projector.")

    #     self.model.classifier.load_state_dict(torch.load(os.path.join(checkpoint_dir, "classifier.bin"), map_location="cpu", weights_only=True))
    #     print("Loaded classification head.")
            
    #     self.model.base_model.language_model.load_adapter(peft_model_id=f"{checkpoint_dir}/lm_adapter")
    #     print("Loaded language model LoRA adapter.")

    # def _load_best_model(self):
    #     best_checkpoint = self.state.best_model_checkpoint
    #     if best_checkpoint is None:
    #         print("No best model checkpoint found.")
    #         return
    #     print(f"Loading best model from {best_checkpoint}")
    #     self.load_model(best_checkpoint)

def extract_first_binary(s: str) -> str:
    """Extract the first occurrence of 0 or 1 in the given string."""
    for char in str(s):
        if char in ['0', '1']:
            return int(char)
    return 0
        
def prepare_loader(summary_type, args, set, processor):
    if set == "train":
        data_path = args.train_data_path
        image_path = args.train_metadata_image_path
    elif set == "dev":
        data_path = args.dev_data_path
        image_path = args.dev_metadata_image_path
    elif set == "test":
        data_path = args.test_data_path
        image_path = args.test_metadata_image_path
    
    data = load_data(data_path, summary_type)
    if args.debug:
        data = data[:70]
    
    dataset = VLM_Dataset(
        args, 
        data, 
        image_path, 
        args.use_cxr_image,
        args.use_rad_report,
        getattr(args, 'use_generated_rad_report', False),
        args.use_discharge_note,
        shuffle=False
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,        
        num_workers=0,  # Set to 0 for deterministic behavior
        collate_fn=custom_data_collator(processor, use_cxr_image=args.use_cxr_image, summary_type=summary_type),
        pin_memory=True
    )
    
    return loader

def load_data(path, summary_type):
    dataset = []
    with open(path, "r") as file:
        for line in file:
            sample = json.loads(line)
            dataset.append({'id' : sample['id'],
                        'label' : sample['label'],
                        'text' : sample[summary_type],
                        'summary_type' : summary_type})
                
    return dataset
        
def compute_metrics_auroc(eval_pred):
    
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
    mapped_state = {}
    for key, value in adapter_state.items():
        parts = key.split('.')
        if 'lora_A' in parts or 'lora_B' in parts:
            weight_index = parts.index('weight')
            parts.insert(weight_index, adapter_name)
        mapped_key = '.'.join(parts)
        mapped_state[mapped_key] = value
    return mapped_state

def compute_ece(labels, pos_probs, n_bins=10):
    """
    Expected Calibration Error (ECE) 계산
    
    Args:
        labels: 실제 라벨 (0 or 1)
        pos_probs: positive class 예측 확률
        n_bins: calibration을 계산할 bin 개수
    
    Returns:
        ece: Expected Calibration Error
    """
    labels = np.asarray(labels).astype(int)
    pos_probs = np.asarray(pos_probs).astype(float)
    
    # bin 경계 생성
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 각 bin에 속하는 샘플 찾기
        in_bin = (pos_probs > bin_lower) & (pos_probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            # bin 내 평균 confidence (예측 확률)
            avg_confidence_in_bin = np.mean(pos_probs[in_bin])
            # bin 내 실제 accuracy (실제 positive 비율)
            avg_accuracy_in_bin = np.mean(labels[in_bin])
            # ECE에 기여
            ece += np.abs(avg_confidence_in_bin - avg_accuracy_in_bin) * prop_in_bin
    
    return float(ece)

def _bootstrap_ci_auc_pr(labels, pos_probs, n_boot=2000, seed=42, stratified=True, alpha=0.05):
    labels = np.asarray(labels).astype(int)
    pos_probs = np.asarray(pos_probs).astype(float)

    rng = np.random.default_rng(seed)

    n = len(labels)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    roc_samples = []
    pr_samples = []

    for _ in range(n_boot):
        if stratified:
            # 클래스 비율 유지하면서 복원추출
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                # 한 클래스면 bootstrap 의미 없음
                break
            boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            boot_idx = np.concatenate([boot_pos, boot_neg])
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

    if len(roc_samples) == 0 or len(pr_samples) == 0:
        return None, None, 0

    lo = 100 * (alpha / 2)
    hi = 100 * (1 - alpha / 2)

    auroc_ci = (float(np.percentile(roc_samples, lo)), float(np.percentile(roc_samples, hi)))
    auprc_ci = (float(np.percentile(pr_samples, lo)), float(np.percentile(pr_samples, hi)))

    return auroc_ci, auprc_ci, int(min(len(roc_samples), len(pr_samples)))

def log_result(args, labels, probs, output_path, set_type,
               compute_ci=True, ci_method="bootstrap",
               n_boot=2000, seed=42):
    os.makedirs(output_path, exist_ok=True)

    labels = np.asarray(labels).astype(int)
    pos_probs = np.asarray([p[1] for p in probs], dtype=float)

    uniq = np.unique(labels)

    if len(uniq) < 2:
        auroc = "NA"
        auprc = "NA"
        auroc_ci = "NA"
        auprc_ci = "NA"
        brier_score = "NA"
        ece = "NA"
        msg = f"[WARN] Only one class present in {set_type} labels: {uniq.tolist()} -> AUROC/AUPRC/CI/Brier/ECE set to NA"
        print(msg)
        n_boot_used = 0
    else:
        auroc = float(roc_auc_score(labels, pos_probs))
        auprc = float(average_precision_score(labels, pos_probs))
        
        # Brier Score 계산
        brier_score = float(brier_score_loss(labels, pos_probs))
        
        # Expected Calibration Error (ECE) 계산
        ece = compute_ece(labels, pos_probs, n_bins=10)

        auroc_ci = "NA"
        auprc_ci = "NA"
        n_boot_used = 0

        if compute_ci:
            if ci_method.lower() == "bootstrap":
                auroc_ci, auprc_ci, n_boot_used = _bootstrap_ci_auc_pr(
                    labels, pos_probs, n_boot=n_boot, seed=seed, stratified=True
                )
                if auroc_ci is None:
                    auroc_ci, auprc_ci = "NA", "NA"
            else:
                raise ValueError("ci_method must be 'bootstrap' (AUPRC CI는 보통 bootstrap만 씁니다).")

    score_file = os.path.join(output_path, "score.txt")
    with open(score_file, "a") as f:
        f.write(f"{args.summary_type}{'_add_pi' if args.use_pi else ''} {set_type} evaluation completed\n")
        f.write(f"Num samples          : {len(labels)}\n")
        f.write(f"Num positives        : {int(labels.sum())}\n")
        f.write(f"Num negatives        : {int((labels == 0).sum())}\n")
        f.write(f"AUROC               : {auroc}\n")
        f.write(f"AUPRC               : {auprc}\n")
        f.write(f"Brier Score         : {brier_score}\n")
        f.write(f"ECE (10 bins)       : {ece}\n")

        if compute_ci:
            f.write(f"CI method           : {ci_method}\n")
            if ci_method.lower() == "bootstrap":
                f.write(f"Bootstrap n         : {n_boot_used}/{n_boot}\n")
            f.write(f"AUROC 95% CI        : {auroc_ci}\n")
            f.write(f"AUPRC 95% CI        : {auprc_ci}\n")

        f.write("\n")

    print(f"Inference complete. Predictions saved to {output_path}")


# Set all random seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


def get_args():
    parser = argparse.ArgumentParser(
        description="LLM zero-shot inference."
    )   
    parser.add_argument(
        "--model_name_or_path", 
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="Hugging Face Hub or local path"
    )
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        default="../trained_models",
        help="Hugging Face Hub or local path"
    )
    parser.add_argument(
        "--train_data_path", 
        type=str, 
        default="../dataset/train_summarization/total_output.jsonl",
        help="Path to a folder or json file"
    )
    parser.add_argument(
        "--dev_data_path", 
        type=str, 
        default="../dataset/dev_summarization/total_output.jsonl",
        help="Path to a folder or json file"
    )
    parser.add_argument(
        "--test_data_path", 
        type=str, 
        default="../dataset/test_summarization/total_output.jsonl",
        help="Path to a folder or json file"
    )
    parser.add_argument(
        '--metadata_path', 
        type=str, 
        default="../dataset/metadata.json",
        help="Path of the dataset to be used"
    )
    parser.add_argument(
        '--train_metadata_image_path', 
        type=str, 
        default="../dataset/train_summarization/full-train-indent-images.json",
        help="Path of the dataset to be used"
    )
    parser.add_argument(
        '--dev_metadata_image_path', 
        type=str, 
        default="../dataset/dev_summarization/full-dev-indent-images.json",
        help="Path of the dataset to be used"
    )
    parser.add_argument(
        '--test_metadata_image_path', 
        type=str, 
        default="../dataset/test_summarization/full-test-indent-images.json",
        help="Path of the dataset to be used"
    )
    parser.add_argument(
        "--base_img_dir", 
        required=True,
        help="Path to a CXR img folder"
    )
    parser.add_argument(
        "--base_rr_dir", 
        required=True,
        help="Path to a radiology report folder"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default='./trained_models', 
        help="Path to save outputs.",
    )
    parser.add_argument(
        "--summary_type", 
        type=str,
        default='plain', 
        help="Discharge note summary type."
    )
    parser.add_argument(
        "--set_name", 
        type=str,
        default='train', 
        help="set type."
    )
    parser.add_argument(
        '--num_epochs', 
        type=int, 
        default=3, 
        help="Use cxr image"
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=5e-5, 
        help="Learing rate"
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=4, 
        help="Use cxr image"
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=42, 
        help="Use cxr image"
    )
    parser.add_argument(
        '--gradient_accumulation_steps', 
        type=int, 
        default=4, 
        help="Use cxr image"
    )
    parser.add_argument(
        '--head_lr', 
        type=float, 
        default=None, 
        help="Learning rate for classifier only (if None, uses --lr)"
    )
    parser.add_argument(
        '--debug', 
        action="store_true",
        help="taining mode"
    )
    parser.add_argument(
        "--finetune", 
        action="store_true",
        help="finetune"
    )
    parser.add_argument(
        "--summarize", 
        action="store_true",
        help="summarize"
    )
    parser.add_argument(
        "--zeroshot", 
        action="store_true",
        help="inference zeroshot"
    )
    parser.add_argument(
        '--wandb', 
        action="store_true",
        help="taining mode"
    )
    parser.add_argument(
        '--use_cxr_image', 
        action="store_true",
        help="Use cxr image"
    )
    parser.add_argument(
        '--use_rad_report', 
        action="store_true",
        help="Use radiology report"
    )
    parser.add_argument(
        '--use_generated_rad_report', 
        action="store_true",
        help="Use generated radiology report (from CheXagent) instead of original"
    )
    parser.add_argument(
        '--use_discharge_note', 
        action="store_true",
        help="Use discharge note during inference"
    )
    parser.add_argument(
        '--use_pi', 
        action="store_true",
        help="Use personal information like gender, age, race"
    )

    return parser.parse_args()

