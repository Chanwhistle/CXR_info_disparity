#!/usr/bin/env python
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import os

import torch
import torch.nn as nn
from peft import LoraConfig
from transformers import (
    AutoProcessor,
    MllamaForConditionalGeneration,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from utils import load_adapter, map_adapter_keys

@dataclass
class ModelConfig:
    num_labels: int = 2
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from peft import LoraConfig
from typing import Optional, Dict

class LlamaMortalityClassificationModel(nn.Module):
    """LLaMA-based model for mortality classification tasks."""
    def __init__(self, args, base_model: PreTrainedModel):
        """
        Initialize the model.
        """
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        # self.custom_config = ModelConfig  # 이 부분은 외부에서 정의되었다고 가정
        self.custom_config = args # args에 config가 포함되어 있다면 이렇게 사용, 아니라면 기존 유지
        self.args = args
        device = next(self.base_model.parameters()).device
        
        # --- [중요] Gradient Checkpointing 오류 해결을 위한 설정 ---
        # 모델의 일부가 freeze 되어도 input embedding에 grad를 흐르게 하여 checkpointing 동작 보장
        if hasattr(self.base_model, "enable_input_require_grads"):
            self.base_model.enable_input_require_grads()
        else:
            # enable_input_require_grads 함수가 없는 경우 수동 설정
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            self.base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # -------------------------------------------------------

        # Determine hidden size
        self.hidden_size = (
            self.base_model.config.hidden_size
            if hasattr(self.base_model.config, "hidden_size")
            else self.base_model.get_input_embeddings().weight.shape[1]
        )
        
        # Initialize classifier
        if not self.args.zeroshot:
            base_dtype = next(self.base_model.parameters()).dtype
            self.classifier = nn.Linear(self.hidden_size, 2).to(device=device, dtype=base_dtype)     
            lora_dropout = getattr(self.custom_config, 'lora_dropout', 0.1)
            self.dropout = nn.Dropout(p=lora_dropout)  
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0], device=device))
            
            # LoRA 및 학습 모듈 설정 호출
            self._setup_lora()
    
    def _setup_lora(self) -> None:
        """Configure LoRA and trainable modules based on modality."""
        
        # 1. 일단 모든 파라미터를 Freeze (기본 상태)
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Classifier는 항상 학습해야 함
        for param in self.classifier.parameters():
            param.requires_grad = True

        # LoRA Config 설정
        lora_config = LoraConfig(
            r=getattr(self.custom_config, 'lora_r', 8),
            lora_alpha=getattr(self.custom_config, 'lora_alpha', 16),
            lora_dropout=getattr(self.custom_config, 'lora_dropout', 0.05),
            target_modules=[
                'q_proj', 'k_proj', 'v_proj', 'o_proj',
                'gate_proj', 'up_proj', 'down_proj',
            ],
            init_lora_weights="gaussian",
            bias="none",
            task_type="SEQ_CLS",
        )

        use_text = (
            getattr(self.args, 'use_discharge_note', False) or 
            getattr(self.args, 'use_rad_report', False) or 
            getattr(self.args, 'use_generated_rad_report', False)
        )
        use_image = getattr(self.args, 'use_cxr_image', False)

        # -------------------------------------------------------
        # Case 1: Text Only (Image X) -> Train LM LoRA
        # -------------------------------------------------------
        if use_text and not use_image:
            print("Configuring for Text Only: Training LM LoRA + classifier")
            self.base_model.language_model.add_adapter(lora_config, adapter_name="language_model_adapter")
            self.base_model.language_model.set_adapter("language_model_adapter")
            # Vision & Projector는 위에서 이미 param.requires_grad=False로 동결됨

        # -------------------------------------------------------
        # Case 2: Image Only (Text X) -> Train Vision LoRA + Projector
        # -------------------------------------------------------
        elif use_image and not use_text:
            print("Configuring for Image Only: Training Vision LoRA + Projector + classifier")
            
            # 1. Vision Encoder에 LoRA 추가
            self.base_model.vision_model.add_adapter(lora_config, adapter_name="vision_model_adapter")
            self.base_model.vision_model.set_adapter("vision_model_adapter")
            
            # 2. Projector 학습 (Full Finetuning)
            for param in self.base_model.multi_modal_projector.parameters():
                param.requires_grad = True
            
            # LM은 동결 상태 유지 (LoRA 추가 안 함)

        # -------------------------------------------------------
        # Case 3: Both (Image + Text) -> Train Vision LoRA + LM LoRA + Projector
        # -------------------------------------------------------
        elif use_text and use_image:
            print("Configuring for Multimodal: Training Vision LoRA + LM LoRA + Projector + classifier")
            
            # 1. LM LoRA
            self.base_model.language_model.add_adapter(lora_config, adapter_name="language_model_adapter")
            self.base_model.language_model.set_adapter("language_model_adapter")
            
            # 2. Vision LoRA
            self.base_model.vision_model.add_adapter(lora_config, adapter_name="vision_model_adapter")
            self.base_model.vision_model.set_adapter("vision_model_adapter")
            
            # 3. Projector 학습
            for param in self.base_model.multi_modal_projector.parameters():
                param.requires_grad = True

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        """Enable gradient checkpointing if supported."""
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            # 경고: base_model 자체에 checkpointing을 켜면, 입력 gradient hook이 필수입니다.
            # __init__에서 처리했으므로 안전합니다.
            self.base_model.gradient_checkpointing_enable(**kwargs)
        else:
            print("Warning: base_model does not support gradient checkpointing.")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        # Remove unnecessary kwargs
        kwargs.pop("num_items_in_batch", None)
        kwargs.pop("ids", None)
        kwargs.pop("summary_type", None)
        
        # Image 사용 안하면 pixel_values 제거
        if not getattr(self.args, 'use_cxr_image', False):
            kwargs.pop("pixel_values", None)
        
        if not self.args.zeroshot:
            outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **kwargs
            )

            hs = outputs.hidden_states[-1]  # (B, T, H)

            if attention_mask is not None:
                idx = attention_mask.long().sum(dim=1) - 1            
                idx = idx.clamp(min=0)                                
                batch_idx = torch.arange(hs.size(0), device=hs.device)
                pooled = hs[batch_idx, idx]                           
            else:
                pooled = hs[:, -1, :]

            # Classifier forward
            # dtype issue 방지: pooled output을 classifier weight 타입으로 캐스팅
            logits = self.classifier(self.dropout(pooled.to(self.classifier.weight.dtype)))
            output_dict = {"logits": logits}

            if labels is not None:
                output_dict["loss"] = self.loss_fn(logits.float(), labels.long())

            # [중요] Loss에 requires_grad가 붙어있는지 확인 (디버깅용, 필요 시 주석 해제)
            # if output_dict["loss"].requires_grad is False:
            #     print("Warning: Loss does not require grad. Check frozen layers.")

            return output_dict

        # (Summarize와 Zeroshot 로직은 그대로 유지)
        if self.args.summarize:     
            outputs = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=8192,
                do_sample=True,
                temperature=0.1,
                **kwargs
            )
            return outputs[0][len(input_ids[0]):]  

        if self.args.zeroshot:     
            outputs = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=8,
                do_sample=False,
                temperature=0.0,
                **kwargs
            )
            return outputs[0][len(input_ids[0]):]        
    
def load_model(
    args,
    model_id: str,
    inference: bool = False,
    attn_implementation: Optional[str] = None,  # e.g. "eager"
):
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.pad_token = "<|finetune_right_pad_id|>"

    extra_kwargs = {}
    if attn_implementation is not None:
        extra_kwargs["attn_implementation"] = attn_implementation

    base_model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",   # ✅ comma
        **extra_kwargs,
    )

    # Wrapper Model 초기화
    model = LlamaMortalityClassificationModel(args, base_model)
    
    if inference:
        # 체크포인트 디렉토리 찾기
        if not os.path.exists(args.checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint_dir}")
            
        # 디렉토리 내부의 하위 폴더(checkpoint-xxx)를 찾거나, 해당 폴더 자체를 사용
        subdirs = [entry.path for entry in os.scandir(args.checkpoint_dir) if entry.is_dir()]
        if subdirs:
            checkpoint_path = Path(subdirs[0]) # 첫 번째 하위 폴더 선택 (보통 최신 or best)
        else:
            checkpoint_path = Path(args.checkpoint_dir) # 하위 폴더 없으면 해당 경로 사용

        print(f"\nLoading weights from {checkpoint_path}...")

        # 1. Classifier 로드 (항상 존재해야 함)
        classifier_path = checkpoint_path / "classifier.bin"
        if classifier_path.exists():
            classifier_state = torch.load(classifier_path, map_location="cpu", weights_only=True)
            model.classifier.load_state_dict(classifier_state, strict=False)
            print("Loaded classifier.")
        else:
            print("Warning: classifier.bin not found!")

        # -------------------------------------------------------
        # 사용 모달리티 확인
        # -------------------------------------------------------
        use_text = (
            getattr(args, 'use_discharge_note', False) or 
            getattr(args, 'use_rad_report', False) or 
            getattr(args, 'use_generated_rad_report', False)
        )
        use_image = getattr(args, 'use_cxr_image', False)

        # 2. Language Model Adapter 로드 (Text 사용 시에만)
        if use_text:
            lm_path = checkpoint_path / "lm_adapter.bin"
            if lm_path.exists():
                # LoRA 로드 유틸리티 함수 사용 가정 (map_adapter_keys, load_adapter)
                lm_adapter_state = map_adapter_keys(torch.load(lm_path, map_location="cpu", weights_only=False), "language_model_adapter")
                current_state_dict = model.base_model.language_model.state_dict()
                load_adapter(current_state_dict, lm_adapter_state)
                model.base_model.language_model.load_state_dict(current_state_dict, strict=False)
                print("Loaded language model LoRA adapter.")
            else:
                print(f"Warning: Text used but {lm_path} not found.")

        # 3. Vision Part 로드 (Image 사용 시에만)
        if use_image:
            # 3-1. Vision Adapter
            vm_adapter_path = checkpoint_path / "vm_adapter.bin"
            if vm_adapter_path.exists():
                vm_adapter_state = map_adapter_keys(torch.load(vm_adapter_path, map_location="cpu", weights_only=False), "vision_model_adapter")
                current_state_dict = model.base_model.vision_model.state_dict()
                load_adapter(current_state_dict, vm_adapter_state)
                model.base_model.vision_model.load_state_dict(current_state_dict, strict=False)
                print("Loaded vision model LoRA adapter.")
            else:
                print(f"Vision adapter not found at {vm_adapter_path} (Might rely on pre-trained if not saved).")

            # 3-2. Multimodal Projector
            multi_modal_projector_path = checkpoint_path / "multi_modal_projector.bin"
            if multi_modal_projector_path.exists():
                multi_modal_projector_state = torch.load(multi_modal_projector_path, map_location="cpu", weights_only=True)
                model.base_model.multi_modal_projector.load_state_dict(multi_modal_projector_state)
                print("Loaded multimodal projector.")
            else:
                print(f"Multimodal projector not found at {multi_modal_projector_path}.")
        
    return model, processor