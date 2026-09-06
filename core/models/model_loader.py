"""Model-family dispatch and shared checkpoint loading."""

import json
import os
from pathlib import Path
from typing import Optional

import torch

from core.models.llama_loader import load_llama_model
from core.models.qwen_loader import load_qwen_model
from core.models.vlm_model import (
    VLMMortalityClassificationModel,
    get_model_components,
)
from core.utils import map_adapter_keys, rank_zero_print


def resolve_model_family(args, model_id: str) -> str:
    """Resolve an explicit or model-id-derived supported model family."""
    family = (getattr(args, "model_family", "auto") or "auto").lower()
    if family == "auto":
        return "qwen" if "qwen" in model_id.lower() else "llama"
    if family not in {"llama", "qwen"}:
        raise ValueError(f"Unsupported model family: {family}. Choose 'llama', 'qwen', or 'auto'.")
    return family


def _model_load_kwargs(args, attn_implementation: Optional[str]):
    kwargs = {}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    if getattr(args, "load_in_4bit", False):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    return kwargs


def _resolve_checkpoint_path(checkpoint_dir: str) -> Path:
    """Resolve an explicit checkpoint or the best saved checkpoint."""
    root = Path(checkpoint_dir)
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    if (root / "classifier.bin").exists():
        return root

    checkpoint_dirs = sorted(
        path for path in root.iterdir()
        if path.is_dir() and path.name.startswith("checkpoint")
    )
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint found in: {checkpoint_dir}")

    best_candidates = []
    for path in checkpoint_dirs:
        state_path = path / "trainer_state.json"
        if not state_path.exists():
            continue
        with open(state_path, "r", encoding="utf-8") as file:
            state = json.load(file)
        best_path = state.get("best_model_checkpoint")
        best_metric = state.get("best_metric")
        if best_path and best_metric is not None:
            candidate = Path(best_path)
            if not candidate.exists():
                candidate = root / candidate.name
            if candidate.exists():
                best_candidates.append((float(best_metric), candidate))

    if best_candidates:
        return max(best_candidates, key=lambda item: item[0])[1]
    if len(checkpoint_dirs) == 1:
        return checkpoint_dirs[0]
    raise FileNotFoundError(
        f"Multiple checkpoints found but no valid best_model_checkpoint in: {checkpoint_dir}"
    )


def _match_adapter_state(adapter_state, target_state):
    """Match adapter keys saved by different PEFT/Transformers versions."""
    matched = {}
    for source_key, value in adapter_state.items():
        candidates = [source_key]
        parts = source_key.split(".")
        while parts and parts[0] in {"base_model", "model"}:
            parts = parts[1:]
            candidates.append(".".join(parts))

        for candidate in candidates:
            target_value = target_state.get(candidate)
            if target_value is not None and target_value.shape == value.shape:
                matched[candidate] = value
                break
    return matched


def _load_adapter_checkpoint(module, path: Path, adapter_name: str, label: str) -> None:
    device = next(module.parameters()).device
    saved_state = torch.load(path, map_location=device, weights_only=False)
    mapped_state = map_adapter_keys(saved_state, adapter_name)
    current_state = module.state_dict()
    matched_state = _match_adapter_state(mapped_state, current_state)
    active_adapter_keys = {
        key for key in current_state
        if f".{adapter_name}.weight" in key
    }
    missing_active_keys = active_adapter_keys - matched_state.keys()

    if not matched_state:
        first_saved = next(iter(saved_state), "<empty checkpoint>")
        raise RuntimeError(
            f"No {label} adapter weights matched {path}. "
            f"First saved key: {first_saved}"
        )
    if missing_active_keys or len(matched_state) != len(saved_state):
        first_missing = next(iter(missing_active_keys), "<none>")
        raise RuntimeError(
            f"Incomplete {label} adapter load from {path}: "
            f"matched {len(matched_state)}/{len(saved_state)} saved tensors and "
            f"{len(matched_state)}/{len(active_adapter_keys)} active tensors. "
            f"First missing active key: {first_missing}"
        )

    for key, value in matched_state.items():
        current_state[key].copy_(value)
    module.load_state_dict(current_state, strict=False, assign=True)
    rank_zero_print(
        f"Loaded {label} LoRA adapter: {len(matched_state)}/{len(saved_state)} tensors."
    )


def _load_checkpoint(model, args) -> None:
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint_dir)
    rank_zero_print(f"\nLoading weights from {checkpoint_path}...")

    classifier_path = checkpoint_path / "classifier.bin"
    if classifier_path.exists():
        base_device = next(model.base_model.parameters()).device
        state = torch.load(classifier_path, map_location=base_device, weights_only=True)
        model.classifier.load_state_dict(state, strict=False, assign=True)
        model.classifier.to(base_device)
        if hasattr(model, "loss_fn"):
            model.loss_fn.weight = model.loss_fn.weight.to(base_device)
        rank_zero_print("Loaded classifier.")
    else:
        rank_zero_print("Warning: classifier.bin not found!")

    use_text = any(
        getattr(args, attr, False)
        for attr in ("use_discharge_note", "use_rad_report", "use_generated_rad_report")
    )
    if use_text:
        lm_path = checkpoint_path / "lm_adapter.bin"
        if lm_path.exists():
            lm, _, _, _ = get_model_components(model.base_model, args.model_family)
            _load_adapter_checkpoint(
                lm,
                lm_path,
                "language_model_adapter",
                "language model",
            )
        else:
            rank_zero_print(f"Warning: Text used but {lm_path} not found.")

    _, vision_model, projector, multimodal = get_model_components(
        model.base_model, args.model_family
    )
    if getattr(args, "use_cxr_image", False) and multimodal:
        vm_path = checkpoint_path / "vm_adapter.bin"
        if vm_path.exists() and vision_model is not None:
            _load_adapter_checkpoint(
                vision_model,
                vm_path,
                "vision_model_adapter",
                "vision model",
            )

        projector_path = checkpoint_path / "multi_modal_projector.bin"
        if projector_path.exists() and projector is not None:
            device = next(projector.parameters()).device
            state = torch.load(projector_path, map_location=device, weights_only=True)
            projector.load_state_dict(state, assign=True)
            rank_zero_print("Loaded multimodal projector.")


def load_model(
    args,
    model_id: str,
    inference: bool = False,
    attn_implementation: Optional[str] = None,
):
    """Load a family-specific base model and wrap it for mortality prediction."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_map = {"": local_rank} if world_size > 1 else "auto"
    model_kwargs = _model_load_kwargs(args, attn_implementation)

    loaders = {
        "llama": load_llama_model,
        "qwen": load_qwen_model,
    }
    family = resolve_model_family(args, model_id)
    args.model_family = family
    base_model, processor = loaders[family](args, model_id, device_map, model_kwargs)
    model = VLMMortalityClassificationModel(args, base_model)

    if inference:
        _load_checkpoint(model, args)

    return model, processor
