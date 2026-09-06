"""Qwen text and vision-language model loading and structure access."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


def get_qwen_lm(base_model: nn.Module) -> nn.Module:
    """Return the Qwen language model."""
    inner_model = getattr(base_model, "model", None)
    language_model = getattr(inner_model, "language_model", None)
    if language_model is not None:
        return language_model

    direct_language_model = getattr(base_model, "language_model", None)
    if direct_language_model is not None:
        return direct_language_model

    return inner_model if inner_model is not None else base_model


def get_qwen_vision_model(base_model: nn.Module):
    """Return the Qwen VL vision model."""
    inner_model = getattr(base_model, "model", None)
    vision_model = getattr(inner_model, "visual", None)
    if vision_model is not None:
        return vision_model
    return getattr(base_model, "visual", None)


def get_qwen_projector(base_model: nn.Module):
    """Return the Qwen VL visual-token projector."""
    vision_model = get_qwen_vision_model(base_model)
    return getattr(vision_model, "merger", None) if vision_model is not None else None


def is_qwen_multimodal(base_model: nn.Module) -> bool:
    """Return whether the Qwen model has a vision model."""
    return get_qwen_vision_model(base_model) is not None


def load_qwen_model(args, model_id: str, device_map, model_kwargs):
    """Load a Qwen base model and its processor/tokenizer."""
    use_image = getattr(args, "use_cxr_image", False)
    is_vision_model = use_image or "vl" in model_id.lower()

    if not is_vision_model:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            **model_kwargs,
        )
        return base_model, tokenizer

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    try:
        from transformers import AutoModelForImageTextToText
        model_cls = AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForVision2Seq
        model_cls = AutoModelForVision2Seq

    base_model = model_cls.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        **model_kwargs,
    )
    return base_model, processor
