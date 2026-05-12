"""
사망률 예측을 위한 Llama 기반 분류 모델 정의 및 로드 유틸리티.

기본 모델: meta-llama/Llama-3.2-11B-Vision-Instruct (멀티모달 Vision Instruct)

핵심 구조:
  - LlamaMortalityClassificationModel: LoRA + 이진 분류 헤드 래퍼
  - load_model(): 모델/프로세서 로드 (Vision 모델 자동 감지 / 텍스트 전용 분기)

LoRA 적용 대상:
  - 텍스트 전용: base_model 직접 (LlamaForCausalLM)
  - 멀티모달: base_model.language_model + vision_model (MllamaForConditionalGeneration)

모델 자동 감지 규칙:
  - model_id에 "vision"이 포함되면 → MllamaForConditionalGeneration + AutoProcessor
  - 그 외 + use_cxr_image=True  → MllamaForConditionalGeneration + AutoProcessor
  - 그 외 + use_cxr_image=False → AutoModelForCausalLM + AutoTokenizer
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import os

import torch
import torch.nn as nn
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from utils import load_adapter, map_adapter_keys


@dataclass
class ModelConfig:
    num_labels: int = 2
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


def _get_lm(base_model: nn.Module) -> nn.Module:
    """언어 모델 컴포넌트 추출: 멀티모달이면 language_model 반환, 텍스트 전용이면 base_model 그대로."""
    return getattr(base_model, 'language_model', base_model)


def _is_multimodal(base_model: nn.Module) -> bool:
    """vision_model 속성 여부로 멀티모달 모델 판별."""
    return hasattr(base_model, 'vision_model') and hasattr(base_model, 'language_model')


class LlamaMortalityClassificationModel(nn.Module):
    """Llama 기반 30일 사망률 이진 분류 모델 (LoRA + 분류 헤드)."""

    def __init__(self, args, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.args = args
        device = next(self.base_model.parameters()).device

        # gradient checkpointing 사용 시 input embedding에 grad 흐름 보장
        if hasattr(self.base_model, "enable_input_require_grads"):
            self.base_model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            self.base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        lm = _get_lm(self.base_model)
        self.hidden_size = (
            lm.config.hidden_size
            if hasattr(lm.config, "hidden_size")
            else lm.get_input_embeddings().weight.shape[1]
        )

        if not self.args.zeroshot:
            base_dtype = next(self.base_model.parameters()).dtype
            self.classifier = nn.Linear(self.hidden_size, 2).to(device=device, dtype=base_dtype)
            lora_dropout = getattr(args, 'lora_dropout', 0.1)
            self.dropout = nn.Dropout(p=lora_dropout)
            # 클래스 불균형 보정: death(1) 가중치 3배
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0], device=device))

            if not getattr(self.args, 'inference', False):
                self._setup_lora()

    def _setup_lora(self) -> None:
        """모달리티에 따라 LoRA 및 학습 파라미터 설정."""

        # 전체 파라미터 freeze
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 분류 헤드는 항상 학습
        for param in self.classifier.parameters():
            param.requires_grad = True

        lora_config = LoraConfig(
            r=getattr(self.args, 'lora_r', 8),
            lora_alpha=getattr(self.args, 'lora_alpha', 16),
            lora_dropout=getattr(self.args, 'lora_dropout', 0.05),
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
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
        lm = _get_lm(self.base_model)
        multimodal = _is_multimodal(self.base_model)

        if use_text and not use_image:
            print("Configuring for Text Only: Training LM LoRA + classifier")
            lm.add_adapter(lora_config, adapter_name="language_model_adapter")
            lm.set_adapter("language_model_adapter")

        elif use_image and not use_text:
            print("Configuring for Image Only: Training Vision LoRA + Projector + classifier")
            if not multimodal:
                raise ValueError("Image-only mode requires a multimodal model (e.g., Llama-3.2-11B-Vision-Instruct)")
            self.base_model.vision_model.add_adapter(lora_config, adapter_name="vision_model_adapter")
            self.base_model.vision_model.set_adapter("vision_model_adapter")
            for param in self.base_model.multi_modal_projector.parameters():
                param.requires_grad = True

        elif use_text and use_image:
            print("Configuring for Multimodal: Training Vision LoRA + LM LoRA + Projector + classifier")
            if not multimodal:
                raise ValueError("Multimodal mode requires a multimodal model (e.g., Llama-3.2-11B-Vision-Instruct)")
            lm.add_adapter(lora_config, adapter_name="language_model_adapter")
            lm.set_adapter("language_model_adapter")
            self.base_model.vision_model.add_adapter(lora_config, adapter_name="vision_model_adapter")
            self.base_model.vision_model.set_adapter("vision_model_adapter")
            for param in self.base_model.multi_modal_projector.parameters():
                param.requires_grad = True

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable(**kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:

        kwargs.pop("num_items_in_batch", None)
        kwargs.pop("ids", None)
        kwargs.pop("summary_type", None)

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

            # 마지막 non-pad 토큰 위치에서 pooling
            if attention_mask is not None:
                idx = attention_mask.long().sum(dim=1) - 1
                idx = idx.clamp(min=0)
                batch_idx = torch.arange(hs.size(0), device=hs.device)
                pooled = hs[batch_idx, idx]
            else:
                pooled = hs[:, -1, :]

            logits = self.classifier(self.dropout(pooled.to(self.classifier.weight)))
            output_dict = {"logits": logits}

            if labels is not None:
                output_dict["loss"] = self.loss_fn(logits.float(), labels.long())

            return output_dict

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
    attn_implementation: Optional[str] = None,
):
    """
    모델과 프로세서/토크나이저를 로드.

    - use_cxr_image=True  → MllamaForConditionalGeneration + AutoProcessor (멀티모달)
    - use_cxr_image=False → AutoModelForCausalLM + AutoTokenizer (텍스트 전용)

    체크포인트 로드 순서 (inference=True):
      1. classifier.bin
      2. lm_adapter.bin (텍스트 사용 시)
      3. vm_adapter.bin + multi_modal_projector.bin (이미지 사용 시)
    """
    use_image = getattr(args, 'use_cxr_image', False)
    extra_kwargs = {}
    if attn_implementation is not None:
        extra_kwargs["attn_implementation"] = attn_implementation

    use_4bit = getattr(args, 'load_in_4bit', False)
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        extra_kwargs["quantization_config"] = bnb_config

    # Vision 모델 여부: use_cxr_image=True 이거나 model_id에 "vision"이 포함된 경우
    is_vision_model = use_image or "vision" in model_id.lower()

    if is_vision_model:
        # 멀티모달: Llama-3.2-11B-Vision-Instruct (MllamaForConditionalGeneration)
        from transformers import MllamaForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.pad_token = "<|finetune_right_pad_id|>"
        base_model = MllamaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", **extra_kwargs
        )
        # 4-bit 로드 시 multi_modal_projector quantization 충돌 방지
        if use_4bit:
            import bitsandbytes as bnb
            proj = base_model.multi_modal_projector
            if isinstance(proj, bnb.nn.Linear4bit):
                proj_device = proj.weight.device
                new_proj = nn.Linear(
                    proj.in_features, proj.out_features,
                    bias=proj.bias is not None,
                    dtype=torch.bfloat16,
                    device=proj_device,
                )
                base_model.multi_modal_projector = new_proj
    else:
        # 텍스트 전용 (non-vision 모델): AutoModelForCausalLM
        processor = AutoTokenizer.from_pretrained(model_id)
        processor.pad_token = processor.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto", **extra_kwargs
        )

    model = LlamaMortalityClassificationModel(args, base_model)

    if inference:
        if not os.path.exists(args.checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint_dir}")

        subdirs = [e.path for e in os.scandir(args.checkpoint_dir) if e.is_dir()]
        checkpoint_path = Path(subdirs[0]) if subdirs else Path(args.checkpoint_dir)
        print(f"\nLoading weights from {checkpoint_path}...")

        # 1. Classifier 로드
        classifier_path = checkpoint_path / "classifier.bin"
        if classifier_path.exists():
            base_device = next(model.base_model.parameters()).device
            classifier_state = torch.load(classifier_path, map_location=base_device, weights_only=True)
            model.classifier.load_state_dict(classifier_state, strict=False, assign=True)
            model.classifier.to(base_device)
            if hasattr(model, 'loss_fn'):
                model.loss_fn.weight = model.loss_fn.weight.to(base_device)
            print("Loaded classifier.")
        else:
            print("Warning: classifier.bin not found!")

        use_text = (
            getattr(args, 'use_discharge_note', False) or
            getattr(args, 'use_rad_report', False) or
            getattr(args, 'use_generated_rad_report', False)
        )

        # 2. Language Model Adapter 로드
        if use_text:
            lm_path = checkpoint_path / "lm_adapter.bin"
            if lm_path.exists():
                lm = _get_lm(model.base_model)
                lm_device = next(lm.parameters()).device
                lm_adapter_state = map_adapter_keys(
                    torch.load(lm_path, map_location=lm_device, weights_only=False),
                    "language_model_adapter"
                )
                current_state_dict = lm.state_dict()
                load_adapter(current_state_dict, lm_adapter_state)
                lm.load_state_dict(current_state_dict, strict=False, assign=True)
                print("Loaded language model LoRA adapter.")
            else:
                print(f"Warning: Text used but {lm_path} not found.")

        # 3. Vision Adapter + Projector 로드 (멀티모달 전용)
        if use_image and _is_multimodal(model.base_model):
            vm_adapter_path = checkpoint_path / "vm_adapter.bin"
            if vm_adapter_path.exists():
                vm_device = next(model.base_model.vision_model.parameters()).device
                vm_adapter_state = map_adapter_keys(
                    torch.load(vm_adapter_path, map_location=vm_device, weights_only=False),
                    "vision_model_adapter"
                )
                current_state_dict = model.base_model.vision_model.state_dict()
                load_adapter(current_state_dict, vm_adapter_state)
                model.base_model.vision_model.load_state_dict(current_state_dict, strict=False, assign=True)
                print("Loaded vision model LoRA adapter.")

            proj_path = checkpoint_path / "multi_modal_projector.bin"
            if proj_path.exists():
                proj_device = next(model.base_model.multi_modal_projector.parameters()).device
                proj_state = torch.load(proj_path, map_location=proj_device, weights_only=True)
                model.base_model.multi_modal_projector.load_state_dict(proj_state, assign=True)
                print("Loaded multimodal projector.")

    return model, processor
