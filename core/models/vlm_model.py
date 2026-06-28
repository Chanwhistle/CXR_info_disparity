"""
Shared Llama/Qwen mortality classifier.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from peft import LoraConfig
from core.utils import rank_zero_print
from core.models.qwen_loader import (
    get_qwen_lm,
    get_qwen_projector,
    get_qwen_vision_model,
    is_qwen_multimodal,
)


def _get_lm(base_model: nn.Module) -> nn.Module:
    """Return the Llama language-model component."""
    language_model = getattr(base_model, "language_model", None)
    if language_model is not None:
        return language_model

    inner_model = getattr(base_model, "model", None)
    nested_language_model = getattr(inner_model, "language_model", None)
    if nested_language_model is not None:
        return nested_language_model
    if inner_model is not None:
        return inner_model
    return base_model


def _is_multimodal(base_model: nn.Module) -> bool:
    """Return True when the model exposes a vision tower."""
    return _get_vision_model(base_model) is not None


def _get_vision_model(base_model: nn.Module) -> Optional[nn.Module]:
    """Return the Llama vision component."""
    for attr in ("vision_model", "visual", "vision_tower"):
        module = getattr(base_model, attr, None)
        if module is not None:
            return module

    inner_model = getattr(base_model, "model", None)
    for attr in ("vision_model", "visual", "vision_tower"):
        module = getattr(inner_model, attr, None)
        if module is not None:
            return module
    return None


def _get_projector(base_model: nn.Module) -> Optional[nn.Module]:
    """Return the Llama multimodal projector."""
    for attr in ("multi_modal_projector", "mm_projector", "visual_projection"):
        module = getattr(base_model, attr, None)
        if module is not None:
            return module

    inner_model = getattr(base_model, "model", None)
    for attr in ("multi_modal_projector", "mm_projector", "visual_projection"):
        module = getattr(inner_model, attr, None)
        if module is not None:
            return module
    return None


def get_model_components(base_model: nn.Module, model_family: str):
    """Return family-specific language, vision, and projector modules."""
    if model_family == "qwen":
        return (
            get_qwen_lm(base_model),
            get_qwen_vision_model(base_model),
            get_qwen_projector(base_model),
            is_qwen_multimodal(base_model),
        )
    return (
        _get_lm(base_model),
        _get_vision_model(base_model),
        _get_projector(base_model),
        _is_multimodal(base_model),
    )


def set_use_cache(model: nn.Module, enabled: bool) -> None:
    """Set use_cache on all model configs."""
    seen = set()
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and id(config) not in seen:
            seen.add(id(config))
            if hasattr(config, "use_cache"):
                config.use_cache = enabled


class VLMMortalityClassificationModel(nn.Module):
    """Binary mortality classifier with LoRA adapters."""

    def __init__(self, args, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.args = args
        device = next(self.base_model.parameters()).device

        # Keep gradients flowing with checkpointing.
        if hasattr(self.base_model, "enable_input_require_grads"):
            self.base_model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            self.base_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        lm, _, _, _ = get_model_components(
            self.base_model, getattr(self.args, "model_family", "llama")
        )
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
            # Upweight the positive class.
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0], device=device))

            if not getattr(self.args, 'inference', False):
                self._setup_lora()

    def _setup_lora(self) -> None:
        """Configure trainable modules by modality."""

        # Freeze the base model.
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Always train the classifier.
        for param in self.classifier.parameters():
            param.requires_grad = True

        model_family = getattr(self.args, "model_family", "llama")
        target_modules = [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
        ]
        if model_family == "qwen":
            target_modules.extend(['qkv', 'proj', 'fc1', 'fc2'])

        lora_config = LoraConfig(
            r=getattr(self.args, 'lora_r', 8),
            lora_alpha=getattr(self.args, 'lora_alpha', 16),
            lora_dropout=getattr(self.args, 'lora_dropout', 0.05),
            target_modules=target_modules,
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
        lm, vision_model, projector, multimodal = get_model_components(
            self.base_model, model_family
        )

        if use_text and not use_image:
            rank_zero_print("Configuring for Text Only: Training LM LoRA + classifier")
            lm.add_adapter(lora_config, adapter_name="language_model_adapter")
            lm.set_adapter("language_model_adapter")

        elif use_image and not use_text:
            rank_zero_print("Configuring for Image Only: Training Vision LoRA + Projector + classifier")
            if not multimodal:
                raise ValueError("Image-only mode requires a multimodal model.")
            vision_model.add_adapter(lora_config, adapter_name="vision_model_adapter")
            vision_model.set_adapter("vision_model_adapter")
            if projector is not None:
                for param in projector.parameters():
                    param.requires_grad = True

        elif use_text and use_image:
            rank_zero_print("Configuring for Multimodal: Training Vision LoRA + LM LoRA + Projector + classifier")
            if not multimodal:
                raise ValueError("Multimodal mode requires a multimodal model.")
            lm.add_adapter(lora_config, adapter_name="language_model_adapter")
            lm.set_adapter("language_model_adapter")
            vision_model.add_adapter(lora_config, adapter_name="vision_model_adapter")
            vision_model.set_adapter("vision_model_adapter")
            if projector is not None:
                for param in projector.parameters():
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

            # Pool the last non-padding token.
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
