"""Llama text and vision-language model loader."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from core.models.vlm_model import get_model_components


def load_llama_model(args, model_id: str, device_map, model_kwargs):
    """Load a Llama base model and its processor/tokenizer."""
    use_image = getattr(args, "use_cxr_image", False)
    is_vision_model = use_image or "vision" in model_id.lower()

    if not is_vision_model:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            **model_kwargs,
        )
        return base_model, tokenizer

    from transformers import MllamaForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.pad_token = "<|finetune_right_pad_id|>"
    base_model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        **model_kwargs,
    )

    if getattr(args, "load_in_4bit", False):
        import bitsandbytes as bnb

        _, _, projector, _ = get_model_components(base_model, "llama")
        if isinstance(projector, bnb.nn.Linear4bit):
            projector_owner = (
                base_model
                if hasattr(base_model, "multi_modal_projector")
                else getattr(base_model, "model", base_model)
            )
            projector_owner.multi_modal_projector = nn.Linear(
                projector.in_features,
                projector.out_features,
                bias=projector.bias is not None,
                dtype=torch.bfloat16,
                device=projector.weight.device,
            )

    return base_model, processor
