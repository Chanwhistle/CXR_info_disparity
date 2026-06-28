"""
Llama/Qwen vision-language model fine-tuning for 30-day mortality prediction.

Trains LoRA adapters and a classification head.

Text-only example:
    python -m train.finetuning \
        --use_discharge_note \
        --base_img_dir ../saved_images_560 \
        --base_rr_dir ../physionet.org/files/mimic-cxr/2.1.0/files \
        --output_path ../trained_models/dn_only
"""

import os
import warnings

import torch
import wandb
from transformers import EarlyStoppingCallback, TrainingArguments, logging as transformers_logging

from core.models import load_model
from core.models.vlm_model import set_use_cache
from core.utils import AdapterOnlyTrainer, load_data, get_args, compute_metrics_auroc, rank_zero_print, set_seed
from core.dataloader import VLM_Dataset, custom_data_collator

warnings.filterwarnings("ignore", message="Could not find a config file in", category=UserWarning)
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True", category=UserWarning)


def train(args):
    train_data = load_data(args.train_data_path, args.summary_type)
    eval_data = load_data(args.dev_data_path, args.summary_type)
    rank_zero_print(f"TOTAL TRAIN DATASET : {len(train_data)}")
    rank_zero_print(f"TOTAL EVAL  DATASET : {len(eval_data)}")

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
    set_use_cache(model, False)

    training_args = TrainingArguments(
        output_dir=args.output_path,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        max_grad_norm=1.0,
        optim="adamw_torch",
        lr_scheduler_type="polynomial",
        warmup_steps=0.1,
        lr_scheduler_kwargs={"power": 0.5},
        report_to=["wandb"] if args.wandb else [],
        save_total_limit=1,
        logging_steps=50,
        dataloader_pin_memory=False,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        # Mllama contains image-conditioned cross-attention layers that are
        # unused for text-only batches even when their LM LoRA is trainable.
        ddp_find_unused_parameters=args.model_family == "llama",
        fp16=False,
        bf16=True,
        metric_for_best_model="auroc",
        greater_is_better=True,
        load_best_model_at_end=True,
    )

    # Right padding is safer for half precision.
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
        processing_class=processor,
        compute_metrics=compute_metrics_auroc,
        use_cxr_image=args.use_cxr_image,
    )

    trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001))
    trainer.train()


if __name__ == "__main__":
    args = get_args()
    if int(os.environ.get("RANK", "0")) != 0:
        transformers_logging.set_verbosity_error()
    os.makedirs(args.output_path, exist_ok=True)
    set_seed(args.seed)
    rank_zero_print(f"Base model           : {args.model_name_or_path}")
    rank_zero_print(f"Model family         : {args.model_family}")
    rank_zero_print(f"Learning rate        : {args.lr}")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps * world_size
    rank_zero_print(f"Per-device batch size: {args.batch_size}")
    rank_zero_print(f"World size           : {world_size}")
    rank_zero_print(f"Effective batch size : {effective_batch_size}")
    rank_zero_print(f"Use CXR image        : {args.use_cxr_image}")
    rank_zero_print(f"Use Radiology note   : {args.use_rad_report}")
    rank_zero_print(f"Use Discharge note   : {args.use_discharge_note}")
    train(args)
