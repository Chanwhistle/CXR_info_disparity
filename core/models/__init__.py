"""Llama/Qwen mortality model public API."""

from core.models.model_loader import load_model
from core.models.vlm_model import VLMMortalityClassificationModel

__all__ = ["VLMMortalityClassificationModel", "load_model"]
