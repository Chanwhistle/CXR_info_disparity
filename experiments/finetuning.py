"""
Llama 3.2 Vision 11B 모델 파인튜닝 스크립트 (30일 사망률 예측).

LoRA + 분류 헤드 방식으로 지정된 모달리티(텍스트/이미지/방사선 보고서 조합)를
학습합니다. 최적 체크포인트는 AUROC 기준으로 선택됩니다.

실행 예시 (텍스트 전용):
    python finetuning.py \
        --use_discharge_note \
        --base_img_dir ../saved_images \
        --base_rr_dir ../physionet.org/files/mimic-cxr/2.1.0/files \
        --output_path ../trained_models/dn_only
"""

import os
import random
import warnings

import torch
import wandb
from transformers import EarlyStoppingCallback, TrainingArguments

from models.llama_model import load_model
from utils import AdapterOnlyTrainer, load_data, get_args, compute_metrics_auroc, set_seed
from dataloader import VLM_Dataset, custom_data_collator

warnings.filterwarnings("ignore", message="Could not find a config file in", category=UserWarning)
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True", category=UserWarning)


def train(args):
    train_data = load_data(args.train_data_path, args.summary_type)
    eval_data = load_data(args.dev_data_path, args.summary_type)
    print(f"TOTAL TRAIN DATASET : {len(train_data)}")
    print(f"TOTAL EVAL  DATASET : {len(eval_data)}")

    if args.debug:
        train_data = train_data[:10]
        eval_data = eval_data[:5]

    train_dataset = VLM_Dataset(
        args=args, data_list=train_data,
        metadata_image_path=args.train_metadata_image_path,
        use_cxr_image=args.use_cxr_image,
        use_rad_report=args.use_rad_report,
        use_generated_rad_report=args.use_generated_rad_report,
        use_discharge_note=args.use_discharge_note,
        shuffle=True
    )
    eval_dataset = VLM_Dataset(
        args=args, data_list=eval_data,
        metadata_image_path=args.dev_metadata_image_path,
        use_cxr_image=args.use_cxr_image,
        use_rad_report=args.use_rad_report,
        use_generated_rad_report=args.use_generated_rad_report,
        use_discharge_note=args.use_discharge_note,
        shuffle=False
    )

    model, processor = load_model(args, model_id=args.model_name_or_path, inference=False)
    model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=args.output_path,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        logging_dir="./logs",
        eval_strategy="epoch",
        save_strategy="epoch",
        max_grad_norm=1.0,
        optim="adamw_hf",
        lr_scheduler_type="polynomial",
        warmup_ratio=0.05,
        lr_scheduler_kwargs={"power": 0.5},
        report_to=["wandb"] if args.wandb else [],
        save_total_limit=1,
        logging_steps=50,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        overwrite_output_dir=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=True,
        fp16=False,
        bf16=True,
        metric_for_best_model="auroc",
        greater_is_better=True,
        load_best_model_at_end=True,
    )

    # 우측 패딩 (half-precision에서 overflow 방지)
    if hasattr(processor, 'tokenizer'):
        processor.tokenizer.padding_side = 'right'
    else:
        processor.padding_side = 'right'

    trainer = AdapterOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=custom_data_collator(processor, use_cxr_image=args.use_cxr_image, summary_type=args.summary_type),
        tokenizer=processor.tokenizer if hasattr(processor, 'tokenizer') else processor,
        compute_metrics=compute_metrics_auroc,
        use_cxr_image=args.use_cxr_image,
        head_lr=getattr(args, 'head_lr', None)
    )

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001))
    trainer.train()


if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_path, exist_ok=True)
    set_seed(random.randint(1, 10000))
    print(f"Base model           : {args.model_name_or_path}")
    print(f"Learning rate        : {args.lr}")
    if getattr(args, 'head_lr', None):
        print(f"Head learning rate   : {args.head_lr}")
    print(f"Batch size           : {args.batch_size * args.gradient_accumulation_steps * torch.cuda.device_count()}")
    print(f"Use CXR image        : {args.use_cxr_image}")
    print(f"Use Radiology note   : {args.use_rad_report}")
    print(f"Use Discharge note   : {args.use_discharge_note}")
    train(args)
