#!/usr/bin/env python
import wandb
import torch
import os
from model import load_model
from trl import SFTConfig
from utils import AdapterOnlySFTTrainer, load_data, get_args, compute_metrics_auroc
from dataloader import VLM_Dataset, custom_data_collator
from transformers import EarlyStoppingCallback, AdamW
import warnings

warnings.filterwarnings(
    "ignore", 
    message="Could not find a config file in", 
    category=UserWarning
)

def train(args):
    train_data = load_data(args.train_data_path, args.summary_type)
    eval_data = load_data(args.dev_data_path, args.summary_type)
    print(f"TOTAL TRAIN DATASET : {len(train_data)}")
    print(f"TOTAL EVAL  DATASET : {len(eval_data)}")
    
    if args.debug:
        train_data = train_data[:20]
        eval_data = eval_data[:5]

    train_dataset = VLM_Dataset(args, 
                                train_data, 
                                args.train_metadata_image_path, 
                                args.use_cxr_image,
                                args.use_rad_report,
                                args.use_discharge_note,
                                shuffle=True)
    eval_dataset = VLM_Dataset(args, 
                               eval_data, 
                               args.dev_metadata_image_path, 
                               args.use_cxr_image,
                               args.use_rad_report,
                               args.use_discharge_note,
                               shuffle=False)

    model, processor = load_model(
        args,
        model_id=args.model_name_or_path, 
        inference=False
        )    
    model.config.use_cache = False
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if (args.wandb) and (local_rank == 0):
        wandb.init(
            project="New-BCH", 
            config=vars(args)
        )

    training_args = SFTConfig(
        output_dir=args.output_path,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.lr,
        logging_dir="./logs",
        eval_strategy="epoch",
        save_strategy="epoch",
        optim="adamw_hf",
        lr_scheduler_type="polynomial",
        warmup_ratio=0.2,
        lr_scheduler_kwargs={"power": 0.5},
        report_to=["wandb"] if args.wandb else [],
        save_total_limit=1,
        logging_steps=10,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        overwrite_output_dir=True,
        remove_unused_columns=False,
        # ddp_find_unused_parameters=False,
        ddp_find_unused_parameters=True,
        fp16=False,
        bf16=True,
        metric_for_best_model="auroc",       
        greater_is_better=True,
        load_best_model_at_end=True,
    )
    
    trainer = AdapterOnlySFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=custom_data_collator(processor, use_cxr_image=args.use_cxr_image, summary_type=args.summary_type),
        processing_class=processor.tokenizer,
        compute_metrics=compute_metrics_auroc,
        args=training_args,
        use_cxr_image=args.use_cxr_image
    )

    trainer.add_callback(
        EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.001
        )
    )
    
    trainer.train()

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.output_path, exist_ok=True)

    print(f"Base model           : {args.model_name_or_path}")
    print(f"Base learning rate   : {args.lr}")
    print(f"Batch size           : {args.batch_size * args.gradient_accumulation_steps * torch.cuda.device_count()}")
    print(f"Use CXR image        : {args.use_cxr_image}")
    print(f"Use Radiology note   : {args.use_rad_report}")
    print(f"Use Discharge note   : {args.use_discharge_note}")
    print(f"LoRA Settting        : {args.lora_setting}")
    train(args)
