import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, MllamaForConditionalGeneration, TrainingArguments, Trainer
from dataloader import load_hash2meta_dict, CXRDecisionTree
from utils import load_jsonl
import gc
from typing import Optional, Dict, Any

# ======================
# Dataset 및 Collator
# ======================
class RadiologyDataset(Dataset):
    def __init__(self, data_list, metadata_image_path):
        self.data_list = data_list
        self.hash2meta = load_hash2meta_dict("../dataset/metadata.json", metadata_image_path)
        self.Decision_tree = CXRDecisionTree()
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        all_img_data_paths = self.hash2meta[item['id']]['metadata_filtered']
        selected_img_data_path = self.Decision_tree.select_best_cxr(all_img_data_paths)
        
        valid_image_path = [os.path.join(
            "/hdd2/chanhwi/mimic-cxr-jpg-2.1.0.physionet.org/files",
            selected_img_data_path[1]
        )]
        valid_report_path = [os.path.join(
            "/hdd0/chanhwi/physionet.org/files/mimic-cxr/2.1.0/files",
            '/'.join(selected_img_data_path[1].split("/")[:3]) + ".txt"
        )]

        sample = {
            "id": item['id'],
            "image": self._load_images(valid_image_path),
            "prompt": self._load_chat_template(valid_image_path),
            "report": self._load_reports(valid_report_path)
        }
        
        return sample
    
    def _load_images(self, image_paths: list) -> list:
        images = []
        for image_path in image_paths:
            if "512_resized" in image_path.lower():
                img_path = image_path
            else:                
                base, ext = image_path.rsplit('.', 1)
                img_path = f"{base}_512_resized.{ext}"
            image = Image.open(img_path).convert('RGB')
            images.append(image)        
        return images
    
    def _load_reports(self, report_paths: list) -> list:
        reports = []
        for report_path in report_paths:
            with open(report_path, "r", encoding='utf-8') as f:
                report = f.read()
            reports.append(report.replace('\n', ' ').strip())
        return reports
          
    def _load_chat_template(self, images):
        messages = [
            {"role": "system", "content": "You are an experienced radiologist generating radiology reports. Your task is to carefully analyze the provided chest X-ray images and produce a comprehensive, accurate report."},
            {"role": "user", "content": [
                *[{"type": "image"} for _ in images],
                {"type": "text", "text": "Please analyze the attached chest X-ray image and generate a comprehensive and detailed radiology report. Clearly describe both normal and abnormal findings."}
            ]}
        ]
        return messages

def create_data_collator(processor):
    def collate_fn(examples: list):
        prompts = [processor.apply_chat_template(ex["prompt"], tokenize=False) for ex in examples]
        reports = [ex["report"][0] for ex in examples]
        images = [ex["image"][0] for ex in examples]
        
        batch = processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32
        )
        reports_batch = processor(
            text=reports,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        batch["reports_input_ids"] = reports_batch['input_ids']
        batch["reports_attention_mask"] = reports_batch['attention_mask']
        batch["ids"] = [ex["id"] for ex in examples]
        batch["raw_reports"] = reports
        batch["raw_prompts"] = prompts
        batch["raw_images"] = images
        return batch
    return collate_fn

def compute_metrics(eval_pred: Dict[str, Any]):
    return {"eval_loss": eval_pred["loss"]}

# ======================
# Loss Function
# ======================
def contrastive_loss(vision_emb, text_emb, temperature=0.07):
    # L2 정규화
    vision_emb = F.normalize(vision_emb, p=2, dim=1)
    text_emb = F.normalize(text_emb, p=2, dim=1)
    
    # 코사인 유사도 기반 로짓
    logits = torch.matmul(vision_emb, text_emb.t()) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2
# ======================
# Custom Model Wrapping Components
# ======================

class VisionAndAdapterOnlySFTTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # vision_encoder 저장 (HuggingFace 모델이라면 save_pretrained 사용)
        vision_output_dir = os.path.join(output_dir, "vision_encoder")
        self.model.vision_encoder.save_pretrained(vision_output_dir)
        
        # adapter 저장 (여기서는 multi_modal_projector를 adapter로 가정)
        adapter_path = os.path.join(output_dir, "multi_modal_projector.bin")
        torch.save(self.model.multi_modal_projector.state_dict(), adapter_path)
        
        print(f"Saved vision model to {vision_output_dir} and adapter to {adapter_path}")

class CustomRadiologyModel(nn.Module):
    def __init__(self, vision_encoder, multi_modal_projector, language_model):
        super(CustomRadiologyModel, self).__init__()
        self.vision_encoder = vision_encoder
        self.multi_modal_projector = multi_modal_projector
        self.language_model = language_model
    
    def forward(self, 
                pixel_values, 
                aspect_ratio_ids, 
                aspect_ratio_mask, 
                reports_input_ids, 
                reports_attention_mask,
                **kwargs
                ):
        # kwargs.pop("num_items_in_batch", None)
        # kwargs.pop("input_ids", None)
        # kwargs.pop("ids", None)
        # kwargs.pop("raw_reports", None)
        # kwargs.pop("raw_images", None)
        # kwargs.pop("raw_prompts", None)

        vision_out = self.vision_encoder(
            pixel_values=pixel_values, 
            aspect_ratio_ids=aspect_ratio_ids, 
            aspect_ratio_mask=aspect_ratio_mask
        )
        vision_features = vision_out.last_hidden_state.squeeze(0)
        vision_features = vision_features.mean(dim=2)  
        vision_emb = vision_features.mean(dim=1)  
        vision_emb = self.multi_modal_projector(vision_emb)

        text_out = self.language_model(
            input_ids=reports_input_ids, 
            attention_mask=reports_attention_mask, 
            output_hidden_states=True
        )
        last_hidden_state = text_out.hidden_states[-1][:, -1, :]
        
        loss = contrastive_loss(vision_emb, last_hidden_state)
        return {"loss": loss}

# ======================
# Argument Parsing
# ======================
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA 3.2 Vision for radiology report alignment with HuggingFace Trainer.")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    parser.add_argument("--train_data_path", type=str, default="../dataset/train_summarization/traindataset_final.jsonl")
    parser.add_argument("--dev_data_path", type=str, default="../dataset/dev_summarization/total_output.jsonl")
    parser.add_argument("--train_metadata_path", type=str, default="../dataset/full-train-indent-images.json")
    parser.add_argument("--dev_metadata_path", type=str, default="../dataset/full-dev-indent-images.json")
    parser.add_argument("--output_path", type=str, default="../finetuned_model/vision_encoder")
    return parser.parse_args()

# ======================
# Main Function: Trainer 사용
# ======================
def main(args):
    # 데이터 로드 및 데이터셋 생성
    train_data = load_jsonl(args.train_data_path)
    dev_data = load_jsonl(args.dev_data_path)
    
    train_dataset = RadiologyDataset(train_data, args.train_metadata_path)
    dev_dataset = RadiologyDataset(dev_data, args.dev_metadata_path)
    
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = 'left'
    processor.tokenizer.pad_token = "<|finetune_right_pad_id|>"
    collate_fn = create_data_collator(processor)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_model = MllamaForConditionalGeneration.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    base_model.tie_weights()
    
    vision_encoder = base_model.vision_model
    multi_modal_projector = base_model.multi_modal_projector
    language_model = base_model.language_model
    
    for param in language_model.parameters():
        param.requires_grad = False
    del base_model
    gc.collect()
    
    # Custom Model 생성
    model = CustomRadiologyModel(vision_encoder, multi_modal_projector, language_model)
    model.to(device)
    
    # TrainingArguments 설정 (여기서 fp16이나 bf16 사용 가능)
    training_args = TrainingArguments(
        output_dir=args.output_path,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        optim="adamw_hf",
        lr_scheduler_type="polynomial",
        warmup_ratio=0.1,
        lr_scheduler_kwargs={"power": 0.5},
        learning_rate=args.lr,
        weight_decay=0.01,
        fp16=False,
        bf16=True,
        logging_dir='./logs',
        logging_steps=10,
        gradient_accumulation_steps=4,
        report_to=["wandb"],
        remove_unused_columns=False,
    )
    
    # HuggingFace Trainer 생성
    trainer = VisionAndAdapterOnlySFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )
    
    # 학습 시작
    trainer.train()
        
    # vision_encoder 저장 (HuggingFace 모델이라면 save_pretrained 사용)
    trainer.model.vision_encoder.save_pretrained(args.output_path)
    
    # adapter 저장 (여기서는 multi_modal_projector를 adapter로 가정)
    adapter_path = os.path.join(args.output_path, "multi_modal_projector.bin")
    torch.save(trainer.model.multi_modal_projector.state_dict(), adapter_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
