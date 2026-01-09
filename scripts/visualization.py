#!/usr/bin/env python

import argparse
import gc
import os
import sys
import json
import math
import time
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from transformers import AutoProcessor, MllamaForConditionalGeneration

# Import from codebase
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from dataloader import VLM_Dataset, load_hash2meta_dict, CXRDecisionTree
from model import load_model
from utils import load_adapter, map_adapter_keys, load_data

# ---------------------------------------------------------------------------
# Utility: Normalize Vision Hidden State (Handle 3D/4D/5D)
# ---------------------------------------------------------------------------

def normalize_vision_hidden_state(hs: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    """
    Normalize vision encoder hidden state to always be 3D [B_flat, T, D].
    Handles tiled/packed modes: 5D [B, Nimg, Ntiles, T, D] or 4D [B, Ntiles, T, D].
    
    Args:
        hs: Hidden state tensor, can be 3D/4D/5D
    
    Returns:
        (hs_flat, meta) where:
        - hs_flat: Always 3D [B_flat, T, D]
        - meta: Dict with structure info
    """
    orig_shape = list(hs.shape)
    ndim = len(orig_shape)
    
    if ndim == 3:
        # [B, T, D] - standard case
        B, T, D = orig_shape
        meta = {
            "orig_shape": orig_shape,
            "layout": "BTD",
            "B": int(B),
            "Nimg": 1,
            "Ntiles": 1,
            "T": int(T),
            "D": int(D),
            "B_flat": int(B)
        }
        return hs, meta
    
    elif ndim == 4:
        # [B, Ntiles, T, D] - tiled mode
        B, Ntiles, T, D = orig_shape
        B_flat = B * Ntiles
        hs_flat = hs.view(B_flat, T, D)
        meta = {
            "orig_shape": orig_shape,
            "layout": "BNTD",
            "B": int(B),
            "Nimg": 1,
            "Ntiles": int(Ntiles),
            "T": int(T),
            "D": int(D),
            "B_flat": int(B_flat)
        }
        return hs_flat, meta
    
    elif ndim == 5:
        # [B, Nimg, Ntiles, T, D] - packed tiled mode
        B, Nimg, Ntiles, T, D = orig_shape
        B_flat = B * Nimg * Ntiles
        hs_flat = hs.view(B_flat, T, D)
        meta = {
            "orig_shape": orig_shape,
            "layout": "BINTD",
            "B": int(B),
            "Nimg": int(Nimg),
            "Ntiles": int(Ntiles),
            "T": int(T),
            "D": int(D),
            "B_flat": int(B_flat)
        }
        return hs_flat, meta
    
    else:
        raise ValueError(
            f"Unsupported hidden state shape: {orig_shape} (expected 3D, 4D, or 5D)"
        )


# ---------------------------------------------------------------------------
# A) Robust Token/Grid Detection
# ---------------------------------------------------------------------------

def detect_vision_tokens_and_grid(
    vision_model: nn.Module,
    pixel_values: torch.Tensor,
    processor: AutoProcessor,
    aspect_ratio_ids: Optional[torch.Tensor] = None,
    aspect_ratio_mask: Optional[torch.Tensor] = None
) -> Tuple[int, int, int, int, int]:
    """
    Robustly detect token indices and patch grid from vision model.
    
    Returns:
        cls_index: Index of CLS token (usually 0)
        n_special: Number of special tokens (CLS + any register tokens)
        num_patches: Number of patch tokens
        grid_h: Height of patch grid
        grid_w: Width of patch grid
    
    Raises:
        ValueError if grid cannot be determined or mismatch detected
    """
    config = vision_model.config
    
    # Try to get from config first
    image_size = None
    patch_size = None
    
    if hasattr(config, 'image_size'):
        image_size = config.image_size
        if isinstance(image_size, (list, tuple)):
            image_size = image_size[0] if len(image_size) > 0 else None
    
    if hasattr(config, 'patch_size'):
        patch_size = config.patch_size
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0] if len(patch_size) > 0 else None
    
    # Infer from pixel_values if not in config
    if image_size is None or patch_size is None:
        if hasattr(processor, 'image_processor'):
            img_proc = processor.image_processor
            if hasattr(img_proc, 'size'):
                size_dict = img_proc.size
                if isinstance(size_dict, dict):
                    image_size = size_dict.get('height') or size_dict.get('shortest_edge')
                elif isinstance(size_dict, (int, list, tuple)):
                    image_size = size_dict if isinstance(size_dict, int) else size_dict[0]
        
        if patch_size is None:
            if image_size:
                if image_size == 224:
                    patch_size = 16
                elif image_size == 336:
                    patch_size = 14
                else:
                    patch_size = 14
    
    # Calculate expected number of patches
    if image_size and patch_size:
        num_patches_expected = (image_size // patch_size) ** 2
        grid_h = grid_w = image_size // patch_size
    else:
        raise ValueError(
            f"Cannot determine patch grid: image_size={image_size}, patch_size={patch_size}. "
            "Please specify in config or processor."
        )
    
    # Try to get sequence length from config first (more reliable)
    seq_len_from_config = None
    if hasattr(config, 'num_patches'):
        seq_len_from_config = config.num_patches + 1  # +1 for CLS
    elif hasattr(config, 'max_position_embeddings'):
        # Sometimes this includes CLS
        seq_len_from_config = config.max_position_embeddings
    elif hasattr(config, 'hidden_size'):
        # Try to infer from image_size/patch_size
        if image_size and patch_size:
            seq_len_from_config = num_patches_expected + 1  # +1 for CLS
    
    # Run a dummy forward to get actual sequence length (fallback if config doesn't work)
    seq_len_from_forward = None
    forward_meta = None
    try:
        with torch.no_grad():
            # Handle 6D pixel_values: [B, Nimg, Ntiles, C, H, W]
            if len(pixel_values.shape) == 6:
                vision_kwargs = {"pixel_values": pixel_values[:1, :, :, :, :, :]}
            else:
                vision_kwargs = {"pixel_values": pixel_values[:1]}
            if aspect_ratio_ids is not None:
                if aspect_ratio_ids.dim() > 0:
                    vision_kwargs["aspect_ratio_ids"] = aspect_ratio_ids[:1]
                else:
                    vision_kwargs["aspect_ratio_ids"] = aspect_ratio_ids
            if aspect_ratio_mask is not None:
                if aspect_ratio_mask.dim() > 0:
                    vision_kwargs["aspect_ratio_mask"] = aspect_ratio_mask[:1]
                else:
                    vision_kwargs["aspect_ratio_mask"] = aspect_ratio_mask
            
            dummy_output = vision_model(**vision_kwargs)
            
            # Extract hidden state
            if hasattr(dummy_output, 'last_hidden_state'):
                hidden_state = dummy_output.last_hidden_state
            elif hasattr(dummy_output, 'hidden_states') and dummy_output.hidden_states:
                # Use last hidden state from tuple
                hidden_state = dummy_output.hidden_states[-1]
            elif isinstance(dummy_output, tuple) and len(dummy_output) > 0:
                hidden_state = dummy_output[0]
                # If first element is also a tuple/BaseModelOutput, recurse
                if hasattr(hidden_state, 'last_hidden_state'):
                    hidden_state = hidden_state.last_hidden_state
                elif hasattr(hidden_state, 'hidden_states') and hidden_state.hidden_states:
                    hidden_state = hidden_state.hidden_states[-1]
            else:
                hidden_state = dummy_output
            
            if not hasattr(hidden_state, 'shape'):
                raise ValueError(f"Cannot extract hidden state from vision model output. Output type: {type(dummy_output)}")
            
            if len(hidden_state.shape) < 2:
                raise ValueError(f"Hidden state shape is invalid: {hidden_state.shape}. Expected [B, T, D] or [B, T]")
            
            # Normalize to handle 3D/4D/5D
            hidden_state_flat, forward_meta = normalize_vision_hidden_state(hidden_state)
            seq_len_from_forward = forward_meta['T']  # Per-tile T
    except Exception as e:
        import traceback
        traceback.print_exc()
        seq_len_from_forward = None
    
    # Use config value if available, otherwise use forward result
    if seq_len_from_config is not None and seq_len_from_config > 1:
        seq_len = seq_len_from_config
    elif seq_len_from_forward is not None and seq_len_from_forward > 1:
        seq_len = seq_len_from_forward
    else:
        # Fallback: use expected patches + 1 (CLS)
        seq_len = num_patches_expected + 1
    
    # Determine special tokens
    n_special = seq_len - num_patches_expected
    cls_index = 0  # CLS is typically at index 0
    
    if n_special < 1:
        raise ValueError(
            f"Invalid token count: seq_len={seq_len}, expected_patches={num_patches_expected}, "
            f"n_special={n_special}. Need at least 1 special token (CLS). "
            f"Please check image_size ({image_size}) and patch_size ({patch_size}) in config."
        )
    
    # Verify grid
    num_patches_actual = seq_len - n_special
    if num_patches_actual != num_patches_expected:
        # Try non-square grids
        for h in range(int(math.sqrt(num_patches_actual)), 0, -1):
            w = num_patches_actual // h
            if h * w == num_patches_actual:
                grid_h, grid_w = h, w
                break
        else:
            raise ValueError(
                f"Patch grid mismatch: actual_patches={num_patches_actual}, "
                f"expected={num_patches_expected} (grid {grid_h}x{grid_w}). "
                f"seq_len={seq_len}, n_special={n_special}"
            )
    
    return cls_index, n_special, num_patches_actual, grid_h, grid_w


# ---------------------------------------------------------------------------
# Vision Encoder Attention Extraction
# ---------------------------------------------------------------------------

def get_attention_via_direct_call(
    model: nn.Module,
    pixel_values: torch.Tensor,
    aspect_ratio_ids: Optional[torch.Tensor] = None,
    aspect_ratio_mask: Optional[torch.Tensor] = None,
    use_captured_attentions: Optional[List] = None
) -> List[torch.Tensor]:
    """
    Directly call vision encoder with output_attentions=True.
    This is the most reliable method if the model supports it.
    """
    vision_model = model.base_model.vision_model
    
    # If we have captured attentions from monkey-patch, use those
    if use_captured_attentions is not None and len(use_captured_attentions) > 0:
        return use_captured_attentions
    
    try:
        with torch.no_grad():
            # Prepare kwargs for vision model
            vision_kwargs = {
                "pixel_values": pixel_values,
                "output_attentions": True,
                "return_dict": True
            }
            
            # Add aspect_ratio_ids and aspect_ratio_mask if provided
            if aspect_ratio_ids is not None:
                vision_kwargs["aspect_ratio_ids"] = aspect_ratio_ids
            if aspect_ratio_mask is not None:
                vision_kwargs["aspect_ratio_mask"] = aspect_ratio_mask
            
            # Try with return_dict=True first
            try:
                outputs = vision_model(**vision_kwargs)
                
                if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    attentions = outputs.attentions
                    
                    if isinstance(attentions, (list, tuple)):
                        # Filter out None values
                        valid_attentions = [attn for attn in attentions if attn is not None and isinstance(attn, torch.Tensor)]
                        if valid_attentions:
                            return [attn.cpu() for attn in valid_attentions]
                    elif isinstance(attentions, torch.Tensor):
                        return [attentions.cpu()]
            except TypeError:
                # Try without return_dict
                vision_kwargs.pop("return_dict", None)
                outputs = vision_model(**vision_kwargs)
            
            # Handle tuple output (common in transformers)
            if isinstance(outputs, tuple):
                # Outputs are typically: (last_hidden_state, pooler_output, attentions, ...)
                for item in outputs:
                    if isinstance(item, (list, tuple)):
                        # Check if this is the attentions list
                        if len(item) > 0 and isinstance(item[0], torch.Tensor):
                            if len(item[0].shape) == 4:  # [B, H, T, T]
                                return [attn.cpu() for attn in item]
                    elif isinstance(item, torch.Tensor) and len(item.shape) == 4:
                        return [item.cpu()]
    except Exception:
        pass
    
    return []


def extract_attention_with_hooks(
    model: nn.Module,
    pixel_values: torch.Tensor,
    aspect_ratio_ids: Optional[torch.Tensor] = None,
    aspect_ratio_mask: Optional[torch.Tensor] = None
) -> List[torch.Tensor]:
    """
    Extract attention using forward hooks on attention modules.
    This is a fallback method that should work for most ViT implementations.
    For MllamaVisionModel, we need to hook the transformer layers.
    """
    vision_model = model.base_model.vision_model
    attention_list = []
    hooks = []
    hooked_names = set()
    
    def create_hook(name):
        def hook(module, input, output):
            # For transformers ViT, attention typically returns (hidden_states, attention_probs)
            if isinstance(output, tuple):
                # Usually: (hidden_states, attention_probs) or just attention_probs
                for idx, out_item in enumerate(output):
                    if isinstance(out_item, torch.Tensor):
                        shape = out_item.shape
                        # Check if it looks like attention: [B, H, T, T] (4D is preferred)
                        if len(shape) == 4:
                            # 4D tensor [B, H, T, T] - this is definitely attention
                            if shape[-1] == shape[-2]:
                                attention_list.append(out_item.detach().cpu())
                                return
                        elif len(shape) == 3:
                            # 3D tensor [B, T, T] - might be attention without heads
                            if shape[-1] == shape[-2]:
                                # Expand to 4D by adding head dimension
                                out_item_expanded = out_item.unsqueeze(1).detach().cpu()  # [B, 1, T, T]
                                attention_list.append(out_item_expanded)
                                return
            elif isinstance(output, torch.Tensor):
                shape = output.shape
                if len(shape) == 4:
                    if shape[-1] == shape[-2]:
                        attention_list.append(output.detach().cpu())
                elif len(shape) == 3:
                    if shape[-1] == shape[-2]:
                        # Expand to 4D
                        output_expanded = output.unsqueeze(1).detach().cpu()
                        attention_list.append(output_expanded)
        return hook
    
    # For MllamaVisionModel, try to find transformer layers
    # Structure: vision_model.transformer.layers or vision_model.global_transformer.layers
    transformer_paths = [
        ('transformer', 'layers'),
        ('global_transformer', 'layers'),
        ('encoder', 'layer'),
    ]
    
    found_layers = False
    for path in transformer_paths:
        try:
            m = vision_model
            for attr in path:
                if hasattr(m, attr):
                    m = getattr(m, attr)
                else:
                    break
            else:
                # Found the layers
                if isinstance(m, (nn.ModuleList, list)) and len(m) > 0:
                    # Hook each layer's attention
                    for i, layer in enumerate(m):
                        # Look for attention submodule in the layer
                        for subname, submodule in layer.named_modules():
                            if 'attention' in subname.lower() and hasattr(submodule, 'forward'):
                                full_name = f"{'.'.join(path)}.{i}.{subname}"
                                if full_name not in hooked_names:
                                    hook = submodule.register_forward_hook(create_hook(full_name))
                                    hooks.append((full_name, hook))
                                    hooked_names.add(full_name)
                    found_layers = True
                    break
        except:
            continue
    
    # If we didn't find transformer layers, try standard patterns
    if not found_layers:
        for name, module in vision_model.named_modules():
            # Skip if we've already hooked a parent module
            if any(hooked in name for hooked in hooked_names):
                continue
                
            # Look for attention modules
            if ('attention' in name.lower() and 'output' not in name.lower()) or 'self_attn' in name:
                if hasattr(module, 'forward') and not any(x in name for x in ['weight', 'bias', 'running', 'num_batches']):
                    hook = module.register_forward_hook(create_hook(name))
                    hooks.append((name, hook))
                    hooked_names.add(name)
    
    if not hooks:
        return []
    
    # Run forward pass
    try:
        with torch.no_grad():
            vision_kwargs = {"pixel_values": pixel_values}
            if aspect_ratio_ids is not None:
                vision_kwargs["aspect_ratio_ids"] = aspect_ratio_ids
            if aspect_ratio_mask is not None:
                vision_kwargs["aspect_ratio_mask"] = aspect_ratio_mask
            _ = vision_model(**vision_kwargs)
    except Exception:
        pass
    
    # Remove hooks
    for name, hook in hooks:
        hook.remove()
    
    # Sort by layer index if we can infer it from names
    if attention_list:
        try:
            layer_indices = []
            for name, _ in hooks[:len(attention_list)]:
                import re
                match = re.search(r'layers?[._](\d+)', name)
                if match:
                    layer_indices.append(int(match.group(1)))
                else:
                    layer_indices.append(len(layer_indices))
            
            if len(layer_indices) == len(attention_list):
                sorted_pairs = sorted(zip(layer_indices, attention_list))
                attention_list = [attn for _, attn in sorted_pairs]
        except:
            pass
    
    return attention_list


# ---------------------------------------------------------------------------
# Attention Visualization Methods
# ---------------------------------------------------------------------------

def get_cls_to_patch_attention(
    attention: torch.Tensor,
    cls_index: int,
    n_special: int
) -> torch.Tensor:
    """
    Extract CLS token attention to all patch tokens using robust indexing.
    
    Args:
        attention: [B, heads, tokens, tokens] attention tensor
        cls_index: Index of CLS token
        n_special: Number of special tokens (patches start after this)
    
    Returns:
        [B, heads, patch_tokens] attention weights
    """
    if attention is None:
        raise ValueError("Attention tensor is None")
    
    if not isinstance(attention, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(attention)}")
    
    if len(attention.shape) != 4:
        raise ValueError(f"Expected 4D tensor [B, H, T, T], got shape {attention.shape}")
    
    B, H, T, _ = attention.shape
    cls_attention = attention[:, :, cls_index, :]  # [B, heads, tokens]
    patch_attention = cls_attention[:, :, n_special:]  # [B, heads, patch_tokens] - robust slicing
    return patch_attention


def compute_last_block_attention(
    attentions: List[torch.Tensor],
    grid_h: int,
    grid_w: int
) -> np.ndarray:
    """
    Compute attention heatmap from the last encoder block.
    Uses robust grid dimensions instead of guessing.
    """
    if not attentions:
        raise ValueError("No attention tensors provided")
    
    # Filter out None values and validate tensors
    valid_attentions = []
    for attn in attentions:
        if attn is not None and isinstance(attn, torch.Tensor):
            # Accept both [B, H, T, T] (full) and [B, H, patch_tokens] (lightweight)
            if len(attn.shape) in [3, 4]:
                valid_attentions.append(attn)
    
    if not valid_attentions:
        raise ValueError("No valid attention tensors found (all are None or invalid shape)")
    
    last_attn = valid_attentions[-1]  # Either [B, H, T, T] or [B, H, patch_tokens]
    
    if len(last_attn.shape) == 3:
        # Already lightweight [B, H, patch_tokens]
        patch_attn = last_attn.mean(dim=1).mean(dim=0)  # [patch_tokens]
    else:
        # Full [B, H, T, T] - this should not happen with lightweight capture
        raise ValueError("Received full attention matrix [B,H,T,T] but expected lightweight [B,H,patch_tokens]")
    
    # Convert to float32 first (BFloat16 not supported by numpy)
    patch_attn_np = patch_attn.float().cpu().numpy()
    num_patches = len(patch_attn_np)
    expected_patches = grid_h * grid_w
    
    # Verify shape matches grid - NO TRUNCATION
    if num_patches != expected_patches:
        raise ValueError(
            f"Patch count mismatch: got {num_patches} patches, "
            f"expected {expected_patches} from grid {grid_h}x{grid_w}. "
            "This indicates incorrect token detection or grid inference."
        )
    
    # Reshape to spatial grid
    heatmap = patch_attn_np.reshape(grid_h, grid_w)
    return heatmap


def compute_attention_rollout(
    attentions: List[torch.Tensor],
    grid_h: int,
    grid_w: int,
    head_fusion: str = "mean"
) -> np.ndarray:
    """
    Compute attention rollout across last N layers (lightweight version).
    Works with pre-sliced CLS-to-patch attention rows [B, H, patch_tokens].
    """
    if not attentions:
        raise ValueError("No attention tensors provided")
    
    # Filter and validate
    valid_attentions = []
    for attn in attentions:
        if attn is not None and isinstance(attn, torch.Tensor):
            if len(attn.shape) == 3:  # [B, H, patch_tokens] - lightweight
                valid_attentions.append(attn)
            elif len(attn.shape) == 4:
                # Full matrix - this shouldn't happen with lightweight capture
                continue
    
    if not valid_attentions:
        raise ValueError("No valid attention tensors found (all are None or invalid shape)")
    
    # Process each layer's CLS-to-patch attention
    rollout = None
    
    for attn in valid_attentions:
        # attn: [B, H, patch_tokens]
        B, H, num_patches = attn.shape
        
        # Fuse heads
        if head_fusion == "mean":
            attn_fused = attn.mean(dim=1)  # [B, patch_tokens]
        elif head_fusion == "max":
            attn_fused = attn.max(dim=1)[0]
        elif head_fusion == "min":
            attn_fused = attn.min(dim=1)[0]
        else:
            attn_fused = attn.mean(dim=1)
        
        # Average over batch and normalize
        attn_fused = attn_fused.mean(dim=0)  # [patch_tokens]
        attn_normalized = attn_fused / (attn_fused.sum() + 1e-8)
        
        # Apply rollout: element-wise multiplication (simpler than full matrix rollout)
        if rollout is None:
            rollout = attn_normalized
        else:
            rollout = rollout * attn_normalized  # Element-wise
            rollout = rollout / (rollout.sum() + 1e-8)  # Renormalize
    
    rollout_np = rollout.float().cpu().numpy()
    
    # Verify shape - NO TRUNCATION
    num_patches = len(rollout_np)
    expected_patches = grid_h * grid_w
    
    if num_patches != expected_patches:
        raise ValueError(
            f"Patch count mismatch: got {num_patches} patches, "
            f"expected {expected_patches} from grid {grid_h}x{grid_w}. "
            "This indicates incorrect token detection or grid inference."
        )
    
    # Reshape to spatial grid
    heatmap = rollout_np.reshape(grid_h, grid_w)
    return heatmap


# ---------------------------------------------------------------------------
# Image Processing and Visualization
# ---------------------------------------------------------------------------

def create_improved_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    grid_h: int,
    grid_w: int,
    alpha: float = 0.35,
    colormap: str = "magma",
    clip_percentiles: Tuple[float, float] = (5.0, 99.0),
    gamma: float = 0.7,
    sigma: float = 5.0,
    border_mask_ratio: float = 0.05,
    lung_mask: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Create improved attention overlay with percentile clipping, smoothing, and better colormap.
    
    Returns:
        Dictionary with keys: 'overlay', 'heatmap_only', 'original'
    """
    img_array = np.array(image).astype(np.float32) / 255.0
    img_h, img_w = img_array.shape[:2]
    
    # Normalize heatmap with percentile clipping
    heatmap_flat = heatmap.flatten()
    valid_mask = np.isfinite(heatmap_flat) & (heatmap_flat > 0)
    
    if valid_mask.sum() == 0:
        heatmap_norm = np.zeros_like(heatmap)
    else:
        valid_values = heatmap_flat[valid_mask]
        clip_low = np.percentile(valid_values, clip_percentiles[0])
        clip_high = np.percentile(valid_values, clip_percentiles[1])
        
        if clip_high <= clip_low:
            clip_high = valid_values.max()
            clip_low = valid_values.min()
        
        # Clip and normalize
        heatmap_clipped = np.clip(heatmap, clip_low, clip_high)
        heatmap_norm = (heatmap_clipped - clip_low) / (clip_high - clip_low + 1e-8)
        
        # Apply gamma correction
        if gamma != 1.0:
            heatmap_norm = np.power(heatmap_norm, gamma)
    
    # Apply border mask (remove edges where artifacts often occur)
    if border_mask_ratio > 0:
        mask_h = int(grid_h * border_mask_ratio)
        mask_w = int(grid_w * border_mask_ratio)
        border_mask = np.ones((grid_h, grid_w), dtype=np.float32)
        border_mask[:mask_h, :] = 0
        border_mask[-mask_h:, :] = 0
        border_mask[:, :mask_w] = 0
        border_mask[:, -mask_w:] = 0
        heatmap_norm = heatmap_norm * border_mask
    
    # Apply lung mask if provided (crude threshold-based mask)
    if lung_mask is not None:
        # Resize lung mask to grid
        lung_mask_resized = F.interpolate(
            torch.from_numpy(lung_mask).float().unsqueeze(0).unsqueeze(0),
            size=(grid_h, grid_w),
            mode="bilinear",
            align_corners=False
        ).squeeze().numpy()
        heatmap_norm = heatmap_norm * lung_mask_resized
    
    # Upsample to image resolution
    heatmap_tensor = torch.from_numpy(heatmap_norm).float().unsqueeze(0).unsqueeze(0)
    heatmap_upsampled = F.interpolate(
        heatmap_tensor,
        size=(img_h, img_w),
        mode="bilinear",
        align_corners=False
    ).squeeze().numpy()
    
    # Apply Gaussian smoothing to reduce noise
    if sigma > 0:
        heatmap_upsampled = ndimage.gaussian_filter(heatmap_upsampled, sigma=sigma)
        # Renormalize after smoothing
        if heatmap_upsampled.max() > 0:
            heatmap_upsampled = heatmap_upsampled / heatmap_upsampled.max()
    
    # Apply colormap
    cmap = plt.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_upsampled)[:, :, :3]  # [H, W, 3] in [0, 1]
    
    # Create overlay
    overlay = (1 - alpha) * img_array + alpha * heatmap_colored
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    
    # Create heatmap-only visualization
    heatmap_only = (heatmap_colored * 255).astype(np.uint8)
    
    # Original image
    original = (img_array * 255).astype(np.uint8)
    
    return {
        'overlay': overlay,
        'heatmap_only': heatmap_only,
        'original': original
    }


# ---------------------------------------------------------------------------
# Data Loading Functions
# ---------------------------------------------------------------------------

def load_sample_by_unique_id(
    unique_id: str,
    args,
    processor: AutoProcessor
) -> Dict:
    """
    Load a single sample by unique_id using the same logic as VLM_Dataset.
    """
    # Load metadata
    hash2meta = load_hash2meta_dict(args.metadata_path, args.metadata_image_path)
    
    if unique_id not in hash2meta:
        raise ValueError(f"Unique ID {unique_id} not found in metadata")
    
    # Create a dummy data item
    item = {
        'id': unique_id,
        'label': None,  # We don't need label for inference
        'text': None,  # Will load from metadata
        'summary_type': args.summary_type
    }
    
    # Load discharge note if available
    # Load from jsonl file using summary_type
    discharge_note = None
    try:
        with open(args.test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                data_item = json.loads(line)
                if data_item.get('id') == unique_id:
                    # Get text based on summary_type (like load_data function)
                    if args.summary_type in data_item:
                        text_data = data_item[args.summary_type]
                        if isinstance(text_data, list):
                            discharge_note = text_data[0]
                        else:
                            discharge_note = text_data
                    # Fallback to 'text' key
                    elif 'text' in data_item:
                        if isinstance(data_item['text'], list):
                            discharge_note = data_item['text'][0]
                        else:
                            discharge_note = data_item['text']
                    break
    except Exception:
        pass
    
    # Load image
    decision_tree = CXRDecisionTree()
    all_img_data_paths = hash2meta[unique_id]['metadata_filtered']
    selected_img_data = decision_tree.select_best_cxr(all_img_data_paths)
    
    if selected_img_data is None:
        raise ValueError(f"No CXR image found for unique_id {unique_id}")
    
    selected_img_data_path = selected_img_data[1]
    
    # Determine split from metadata_image_path
    metadata_path_lower = args.metadata_image_path.lower()
    if 'train' in metadata_path_lower:
        split = 'train'
    elif 'dev' in metadata_path_lower or 'val' in metadata_path_lower:
        split = 'dev'
    elif 'test' in metadata_path_lower:
        split = 'test'
    else:
        split = 'test'  # default
    
    # Load image
    image_path = selected_img_data_path.split("/")[-1]
    name, extension = image_path.split(".")
    if "_512_resized" in name:
        real_image_path = os.path.join(args.base_img_dir, split, image_path)
    else:
        real_image_path = os.path.join(args.base_img_dir, split, f"{name}_512_resized.{extension}")
    
    if not os.path.exists(real_image_path):
        raise FileNotFoundError(f"Image not found: {real_image_path}")
    
    img = Image.open(real_image_path).convert("RGB")
    
    # Load radiology reports
    rad_report = None
    generated_rad_report = None
    
    if selected_img_data_path:
        path_parts = selected_img_data_path.split("/")[:3]
        if len(path_parts) == 3:
            rr_relative_path = '/'.join(path_parts) + ".txt"
            original_report_path = os.path.join(args.base_rr_dir, rr_relative_path)
            
            report_dir = '/'.join(path_parts[:-1])
            report_filename = path_parts[-1] + ".txt"
            generated_rr_relative_path = f"{report_dir}/generated_{report_filename}"
            generated_report_path = os.path.join(args.base_rr_dir, generated_rr_relative_path)
            
            if os.path.exists(original_report_path):
                with open(original_report_path, "r", encoding='utf-8') as f:
                    rad_report = f.read().replace('\n', ' ').strip()
            
            if os.path.exists(generated_report_path):
                with open(generated_report_path, "r", encoding='utf-8') as f:
                    generated_rad_report = f.read().replace('\n', ' ').strip()
    
    # Prepare sample
    sample = {
        'id': unique_id,
        'discharge_note': discharge_note,
        'radiology_report': rad_report,
        'generated_radiology_report': generated_rad_report,
        'image': img,
        'image_path': real_image_path,
        'summary_type': args.summary_type
    }
    
    return sample


# ---------------------------------------------------------------------------
# Gradient Attribution Hook
# ---------------------------------------------------------------------------

class PatchAttributionHook:
    """Hook to capture patch embeddings and compute gradient attribution."""
    
    def __init__(self):
        self.activations = None
        self.gradients = None
        self.fwd_handle = None
        self.bwd_handle = None
    
    def forward_hook(self, module, input, output):
        """Capture activations."""
        if isinstance(output, (tuple, list)):
            output = output[0]
        
        self.activations = output
        if self.activations is not None and self.activations.requires_grad:
            try:
                self.activations.retain_grad()
            except RuntimeError:
                pass
    
    def backward_hook(self, module, grad_input, grad_output):
        """Capture gradients."""
        if grad_output is not None and len(grad_output) > 0:
            self.gradients = grad_output[0]
    
    def register(self, module):
        """Register hooks on module."""
        self.fwd_handle = module.register_forward_hook(self.forward_hook)
        self.bwd_handle = module.register_full_backward_hook(self.backward_hook)
    
    def remove(self):
        """Remove hooks."""
        if self.fwd_handle:
            self.fwd_handle.remove()
        if self.bwd_handle:
            self.bwd_handle.remove()
        # Clear internal state
        self.activations = None
        self.gradients = None
    
    def cleanup(self):
        """Explicitly cleanup GPU memory."""
        if self.activations is not None and isinstance(self.activations, torch.Tensor):
            if self.activations.is_cuda:
                del self.activations
        if self.gradients is not None and isinstance(self.gradients, torch.Tensor):
            if self.gradients.is_cuda:
                del self.gradients
        self.activations = None
        self.gradients = None


def find_patch_embedding_layer(vision_model: nn.Module) -> Optional[nn.Module]:
    """Find the best layer to hook into for patch embeddings."""
    # Try to find patch embedding or first transformer layer output
    if hasattr(vision_model, 'embeddings'):
        emb = vision_model.embeddings
        if hasattr(emb, 'patch_embedding'):
            return emb.patch_embedding
        # If no patch_embedding, try to hook first transformer layer
        if hasattr(vision_model, 'transformer') and hasattr(vision_model.transformer, 'layers'):
            if len(vision_model.transformer.layers) > 0:
                return vision_model.transformer.layers[0]
    
    if hasattr(vision_model, 'transformer') and hasattr(vision_model.transformer, 'layers'):
        if len(vision_model.transformer.layers) > 0:
            return vision_model.transformer.layers[0]
    
    return None


# ===========================================================================
# 수정된 함수 1: Gradient Attribution (메모리 최적화 적용)
# ===========================================================================

def compute_gradient_attribution(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    target_class: int = 1,
    grid_h: int = 40,
    grid_w: int = 40,
    cls_index: int = 0,
    n_special: int = 1,
    num_patches: int = 1600,
    representative_tile: int = 0,
    vision_meta: Optional[Dict] = None
) -> np.ndarray:
    """
    Compute gradient-based attribution using AMP & Flash Attention.
    Re-enables Flash Attention to fix OOM caused by Eager Attention.
    """
    from torch.cuda.amp import autocast
    import gc

    # -----------------------------------------------------------------------
    # 1. 메모리 정리 (가장 먼저 수행)
    # -----------------------------------------------------------------------
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # -----------------------------------------------------------------------
    # 2. Flash Attention / SDPA 재활성화 (OOM 해결의 핵심)
    # -----------------------------------------------------------------------
    # 시각화를 위해 꺼뒀던 효율적인 Attention을 역전파를 위해 다시 켭니다.
    # Gradient 계산에는 Attention Weight값 자체가 필요 없으므로 Flash Attention이 유리합니다.
    prev_flash = torch.backends.cuda.flash_sdp_enabled()
    prev_mem_eff = torch.backends.cuda.mem_efficient_sdp_enabled()
    prev_math = torch.backends.cuda.math_sdp_enabled()

    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False) # Eager(Math) 끄기
    
    # 모델 Config도 임시로 변경 (지원하는 경우)
    vision_config = model.base_model.vision_model.config
    prev_attn_impl = getattr(vision_config, "attn_implementation", None)
    if hasattr(vision_config, "attn_implementation"):
        # sdpa나 flash_attention_2로 변경 시도
        vision_config.attn_implementation = "sdpa"

    # -----------------------------------------------------------------------
    # 3. 모델 파라미터 얼리기 & Checkpointing
    # -----------------------------------------------------------------------
    for param in model.parameters():
        param.requires_grad = False
        
    model.eval()
    model.zero_grad(set_to_none=True)
    
    chkpt_enabled = False
    try:
        # Vision Model에 대해 Checkpointing 활성화 시도
        if hasattr(model.base_model.vision_model, "gradient_checkpointing_enable"):
             model.base_model.vision_model.gradient_checkpointing_enable()
             chkpt_enabled = True
        # 전체 모델에 대해 시도
        elif hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
            chkpt_enabled = True
    except Exception:
        pass

    vision_model = model.base_model.vision_model
    patch_layer = find_patch_embedding_layer(vision_model)
    
    if patch_layer is None:
        # 복구 및 종료
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(prev_flash)
            torch.backends.cuda.enable_mem_efficient_sdp(prev_mem_eff)
            torch.backends.cuda.enable_math_sdp(prev_math)
        raise ValueError("Could not find patch embedding layer")
    
    hook = PatchAttributionHook()
    hook.register(patch_layer)
    
    try:
        # Input 준비
        pixel_values = batch["pixel_values"].to(device)
        pixel_values = pixel_values.clone().detach().requires_grad_(True)
        
        batch_grad = {}
        for k, v in batch.items():
            if k == "pixel_values":
                batch_grad[k] = pixel_values
            elif isinstance(v, torch.Tensor):
                batch_grad[k] = v.to(device)
            else:
                batch_grad[k] = v
        
        # -----------------------------------------------------------------------
        # 4. Forward & Backward (with Autocast)
        # -----------------------------------------------------------------------
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        with torch.set_grad_enabled(True):
             with autocast(dtype=dtype):
                 outputs = model(**batch_grad)
                 logits = outputs["logits"]
                 target = logits[:, target_class].sum()
             
             target.backward(retain_graph=False)
        
        if hook.activations is None or hook.gradients is None:
            raise RuntimeError("Gradients not captured")
        
        # 결과 CPU 이동 및 GPU 메모리 즉시 해제
        act = hook.activations.detach().cpu().float()
        grad = hook.gradients.detach().cpu().float()
        
        del batch_grad, outputs, logits, target, pixel_values
        hook.remove()
        hook.cleanup()
        
        # 모델 설정 복구 전에 먼저 메모리 비우기
        model.zero_grad(set_to_none=True)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # -----------------------------------------------------------------------
        # 5. 결과 처리 (CPU)
        # -----------------------------------------------------------------------
        if act.dim() == 3:
            B_flat, T, D = act.shape
            if T > num_patches + n_special:
                act_patches = act[:, cls_index+1:cls_index+1+num_patches, :]
                grad_patches = grad[:, cls_index+1:cls_index+1+num_patches, :]
            else:
                act_patches = act[:, 1:, :]
                grad_patches = grad[:, 1:, :]
        elif act.dim() == 5:
            B, Nimg, Ntiles, T, D = act.shape
            B_flat = B * Nimg * Ntiles
            act = act.view(B_flat, T, D)
            grad = grad.view(B_flat, T, D)
            if T > num_patches + n_special:
                act_patches = act[:, cls_index+1:cls_index+1+num_patches, :]
                grad_patches = grad[:, cls_index+1:cls_index+1+num_patches, :]
            else:
                act_patches = act[:, 1:, :]
                grad_patches = grad[:, 1:, :]
        else:
            act_patches = act
            grad_patches = grad

        attribution = (grad_patches * act_patches).sum(dim=-1).abs()
        
        # Tile 처리
        if vision_meta and vision_meta.get('Ntiles', 1) > 1:
            per_tile_patches = num_patches
            tile_start = representative_tile * per_tile_patches
            tile_end = (representative_tile + 1) * per_tile_patches
            
            if attribution.shape[1] >= tile_end:
                attribution = attribution[:, tile_start:tile_end]
            else:
                attribution = attribution[:, :per_tile_patches]
        
        # Grid Reshape
        if attribution.shape[1] == grid_h * grid_w:
            attribution = attribution.view(-1, grid_h, grid_w)
        else:
            side = int(np.sqrt(attribution.shape[1]))
            attribution = attribution.view(-1, side, side)

        return attribution[0].numpy()
    
    except Exception:
        return np.zeros((grid_h, grid_w))
        
    finally:
        # -----------------------------------------------------------------------
        # 6. 설정 복구 (원래대로 Eager Attention 등으로 되돌리기)
        # -----------------------------------------------------------------------
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(prev_flash)
            torch.backends.cuda.enable_mem_efficient_sdp(prev_mem_eff)
            torch.backends.cuda.enable_math_sdp(prev_math)
        
        if hasattr(vision_config, "attn_implementation") and prev_attn_impl:
            vision_config.attn_implementation = prev_attn_impl

        # Checkpointing 끄기
        if chkpt_enabled:
            try:
                if hasattr(model.base_model.vision_model, "gradient_checkpointing_disable"):
                    model.base_model.vision_model.gradient_checkpointing_disable()
                elif hasattr(model, "gradient_checkpointing_disable"):
                    model.gradient_checkpointing_disable()
            except:
                pass
        
        model.zero_grad(set_to_none=True)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


# ===========================================================================
# 수정된 함수 2: Occlusion Sensitivity (배치 사이즈 1 강제)
# ===========================================================================

def compute_occlusion_sensitivity(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    processor: AutoProcessor,
    sample: Dict,
    target_class: int = 1,
    grid_h: int = 40,
    grid_w: int = 40,
    representative_tile: int = 0,
    vision_meta: Optional[Dict] = None,
    batch_size: int = 16,
    mask_value: float = 0.0,
    occlusion_mode: str = "patch",
    stride: int = 2
) -> Tuple[np.ndarray, float]:
    """
    Compute occlusion sensitivity with Stride and Debug Stats.
    """
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    with torch.no_grad():
        base_outputs = model(**batch)
        base_logits = base_outputs["logits"]
        base_death_logit = float(base_logits[0, target_class].item())
    
    pixel_values = batch["pixel_values"]
    if len(pixel_values.shape) == 6:
        _, _, _, _, H, W = pixel_values.shape
        patch_h = H // grid_h
        patch_w = W // grid_w
    elif len(pixel_values.shape) == 4:
        _, _, H, W = pixel_values.shape
        patch_h = H // grid_h
        patch_w = W // grid_w
        
    block_h = patch_h * stride
    block_w = patch_w * stride
    step_h = grid_h // stride
    step_w = grid_w // stride
    total_steps = step_h * step_w
    
    small_heatmap = np.zeros((step_h, step_w), dtype=np.float32)
    
    from tqdm import tqdm
    iterator = tqdm(range(step_h * step_w), desc=f"Occlusion (Stride={stride})")
    
    idx = 0
    for r in range(step_h):
        for c in range(step_w):
            row_start = r * block_h
            row_end = min((r + 1) * block_h, H)
            col_start = c * block_w
            col_end = min((c + 1) * block_w, W)
            
            batch_masked = {}
            for k, v in batch.items():
                if k == "pixel_values":
                    pv_masked = v.clone()
                    if len(pv_masked.shape) == 6:
                        pv_masked[:, :, representative_tile, :, row_start:row_end, col_start:col_end] = mask_value
                    else:
                        pv_masked[:, :, row_start:row_end, col_start:col_end] = mask_value
                    batch_masked[k] = pv_masked.to(device)
                elif isinstance(v, torch.Tensor):
                    batch_masked[k] = v.to(device)
                else:
                    batch_masked[k] = v
            
            with torch.no_grad():
                outputs_masked = model(**batch_masked)
                masked_logit = outputs_masked["logits"][0, target_class].item()
                # Score = 원래 점수 - 가렸을 때 점수 (클수록 해당 영역이 중요함)
                small_heatmap[r, c] = base_death_logit - masked_logit
            
            del batch_masked, outputs_masked
            if idx % 50 == 0:
                torch.cuda.empty_cache()
            
            idx += 1
            iterator.update(1)

    import scipy.ndimage
    zoom_factor = (grid_h / step_h, grid_w / step_w)
    # order=0 (Nearest)을 사용해야 블록 모양이 유지되어 흐려짐 방지
    occlusion_heatmap = scipy.ndimage.zoom(small_heatmap, zoom_factor, order=0)
    
    del pixel_values
    torch.cuda.empty_cache()
    
    return occlusion_heatmap, base_death_logit

# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------

def validate_token_structure(
    vision_model: nn.Module,
    pixel_values: torch.Tensor,
    processor: AutoProcessor,
    aspect_ratio_ids: Optional[torch.Tensor],
    aspect_ratio_mask: Optional[torch.Tensor],
    expected_cls_index: int,
    expected_n_special: int,
    model_name: str = "Model"
) -> Tuple[int, int, int, int, int]:
    """Validate that token structure matches expectations."""
    cls_index, n_special, num_patches, grid_h, grid_w = detect_vision_tokens_and_grid(
        vision_model, pixel_values, processor, aspect_ratio_ids, aspect_ratio_mask
    )
    
    if cls_index != expected_cls_index:
        raise ValueError(
            f"{model_name}: CLS index mismatch: {cls_index} != {expected_cls_index}. "
            "This indicates different token layouts between models."
        )
    if n_special != expected_n_special:
        raise ValueError(
            f"{model_name}: n_special mismatch: {n_special} != {expected_n_special}. "
            "This indicates different special token counts between models."
        )
    
    return cls_index, n_special, num_patches, grid_h, grid_w


def validate_attention_masks(
    mask_1: Optional[torch.Tensor],
    mask_2: Optional[torch.Tensor],
    name: str = "mask"
) -> bool:
    """Validate that attention masks are identical."""
    if mask_1 is None and mask_2 is None:
        return True
    if mask_1 is None or mask_2 is None:
        return False
    
    if not torch.equal(mask_1, mask_2):
        return False
    
    return True


def compare_attention_heatmaps(
    heatmap_1: np.ndarray,
    heatmap_2: np.ndarray,
    threshold: float = 1e-5
) -> Dict[str, float]:
    """Compare two attention heatmaps and return statistics."""
    h1 = np.array(heatmap_1, dtype=np.float32)
    h2 = np.array(heatmap_2, dtype=np.float32)
    
    if h1.shape != h2.shape:
        raise ValueError(f"Heatmap shape mismatch: {h1.shape} != {h2.shape}")
    
    diff = np.abs(h1 - h2)
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    std_diff = float(diff.std())
    
    # Compute correlation
    h1_flat = h1.flatten()
    h2_flat = h2.flatten()
    correlation = float(np.corrcoef(h1_flat, h2_flat)[0, 1])
    
    stats = {
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "std_abs_diff": std_diff,
        "correlation": correlation,
        "is_identical": max_diff < threshold
    }
    
    return stats


# ---------------------------------------------------------------------------
# Comparison Visualization
# ---------------------------------------------------------------------------

# ===========================================================================
# 수정된 함수: Visualization (Occlusion 강제 증폭 로직 추가)
# ===========================================================================

def create_single_model_comparison_figure(
    original_image: Image.Image,
    attention_heatmap: np.ndarray,
    gradient_heatmap: Optional[np.ndarray],
    occlusion_heatmap: Optional[np.ndarray],
    pred: int,
    prob: float,
    output_path: str,
    unique_id: str,
    args
) -> None:
    """
    Create 4-panel comparison figure.
    Includes FIX for invisible Occlusion Heatmaps (Auto-scaling).
    """
    img_array = np.array(original_image).astype(np.float32) / 255.0
    img_h, img_w = img_array.shape[:2]
    
    # Parameters
    viz_clip_low = max(args.clip_low, 50.0)
    viz_clip_high = min(args.clip_high, 99.5)
    viz_gamma = max(0.9, min(args.gamma, 1.0))
    viz_sigma = min(args.sigma, 1.0)
    
    # 1. Attention Normalization
    attn_norm = normalize_for_vis_comparison(attention_heatmap, viz_clip_low, viz_clip_high, viz_gamma)
    attn_h, attn_w = attn_norm.shape
    if attn_h != img_h or attn_w != img_w:
        attn_vis = ndimage.zoom(attn_norm, (img_h / attn_h, img_w / attn_w), order=1)
    else:
        attn_vis = attn_norm
    if args.sigma > 0:
        attn_vis = ndimage.gaussian_filter(attn_vis, sigma=args.sigma)
    
    # 2. Gradient Normalization (if available)
    grad_vis = None
    if gradient_heatmap is not None:
        grad_norm = normalize_for_vis_comparison(gradient_heatmap, viz_clip_low, viz_clip_high, viz_gamma)
        grad_h, grad_w = grad_norm.shape
        if grad_h != img_h or grad_w != img_w:
            grad_vis = ndimage.zoom(grad_norm, (img_h / grad_h, img_w / grad_w), order=1)
        else:
            grad_vis = grad_norm
        if args.sigma > 0:
            grad_vis = ndimage.gaussian_filter(grad_vis, sigma=args.sigma)
    
    occ_vis = None
    if occlusion_heatmap is not None:
        occ_pos = np.maximum(occlusion_heatmap, 0)
        occ_max = occ_pos.max()
        
        if occ_max > 1e-8:
            occ_norm = occ_pos / occ_max
        else:
            occ_norm = np.zeros_like(occ_pos)

        occ_norm = np.power(occ_norm, 0.6)
        
        occ_h, occ_w = occ_norm.shape
        if occ_h != img_h or occ_w != img_w:
            occ_vis = ndimage.zoom(occ_norm, (img_h / occ_h, img_w / occ_w), order=0)
        else:
            occ_vis = occ_norm
            
    # Apply colormap
    cmap = plt.get_cmap(args.colormap)
    attn_colored = cmap(attn_vis)[:, :, :3]
    if grad_vis is not None:
        grad_colored = cmap(grad_vis)[:, :, :3]
    if occ_vis is not None:
        occ_colored = cmap(occ_vis)[:, :, :3]
    
    # Create overlays
    attn_overlay = (1 - args.alpha) * img_array + args.alpha * attn_colored
    attn_overlay = np.clip(attn_overlay * 255, 0, 255).astype(np.uint8)
    
    if grad_vis is not None:
        grad_overlay = (1 - args.alpha) * img_array + args.alpha * grad_colored
        grad_overlay = np.clip(grad_overlay * 255, 0, 255).astype(np.uint8)
    else:
        grad_overlay = None
    
    if occ_vis is not None:
        occ_overlay = (1 - args.alpha) * img_array + args.alpha * occ_colored
        occ_overlay = np.clip(occ_overlay * 255, 0, 255).astype(np.uint8)
    else:
        occ_overlay = None
    
    # Create figure
    num_panels = 1 + (1 if gradient_heatmap is not None else 0) + (1 if occlusion_heatmap is not None else 0) + 1
    fig, axes = plt.subplots(1, num_panels, figsize=(5*num_panels, 5))
    if num_panels == 1:
        axes = [axes]
    
    # Panel 1: Original
    axes[0].imshow(img_array)
    axes[0].set_title(f"Original Image\n{unique_id}", fontsize=8)
    axes[0].axis("off")
    
    # Panel 2: Attention
    axes[1].imshow(attn_overlay)
    axes[1].set_title(f"Attention Map", fontsize=8)
    axes[1].axis("off")
    
    panel_idx = 2
    # Panel 3: Gradient
    if grad_overlay is not None:
        axes[panel_idx].imshow(grad_overlay)
        axes[panel_idx].set_title("Gradient Attribution", fontsize=8)
        axes[panel_idx].axis("off")
        panel_idx += 1
    
    # Panel 4: Occlusion
    if occ_overlay is not None:
        axes[panel_idx].imshow(occ_overlay)
        
        axes[panel_idx].set_title(f"Occlusion Sensitivity", fontsize=8)
        axes[panel_idx].axis("off")
        panel_idx += 1
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    


def create_comparison_figure(
    original_image: Image.Image,
    heatmap_1: np.ndarray,
    heatmap_2: np.ndarray,
    diff_map: np.ndarray,
    pred_1: int,
    prob_1: float,
    pred_2: int,
    prob_2: float,
    valid_tiles_1: List[int],
    valid_tiles_2: List[int],
    rep_tile_1: int,
    rep_tile_2: int,
    output_path: str,
    unique_id: str,
    model_name_1: str,
    model_name_2: str,
    args
) -> None:
    """Create 4-panel comparison figure for attention heatmaps."""
    img_array = np.array(original_image).astype(np.float32) / 255.0
    img_h, img_w = img_array.shape[:2]
    
    # Normalize heatmaps for visualization
    heatmap_1_norm = normalize_for_vis_comparison(heatmap_1, args.clip_low, args.clip_high, args.gamma)
    heatmap_2_norm = normalize_for_vis_comparison(heatmap_2, args.clip_low, args.clip_high, args.gamma)
    
    # Upsample to image size
    attr_h, attr_w = heatmap_1_norm.shape
    if attr_h != img_h or attr_w != img_w:
        heatmap_1_vis = ndimage.zoom(heatmap_1_norm, (img_h / attr_h, img_w / attr_w), order=1)
        heatmap_2_vis = ndimage.zoom(heatmap_2_norm, (img_h / attr_h, img_w / attr_w), order=1)
    else:
        heatmap_1_vis = heatmap_1_norm
        heatmap_2_vis = heatmap_2_norm
    
    # Gaussian smoothing
    if args.sigma > 0:
        heatmap_1_vis = ndimage.gaussian_filter(heatmap_1_vis, sigma=args.sigma)
        heatmap_2_vis = ndimage.gaussian_filter(heatmap_2_vis, sigma=args.sigma)
    
    # Normalize difference map
    diff_abs = np.abs(diff_map)
    s = np.percentile(diff_abs, args.diff_clip) if hasattr(args, 'diff_clip') else np.percentile(diff_abs, 99.0)
    diff_norm = np.clip(diff_map / (s + 1e-8), -1, 1)
    diff_vis = ndimage.zoom(diff_norm, (img_h / attr_h, img_w / attr_w), order=1) if attr_h != img_h else diff_norm
    if args.sigma > 0:
        diff_vis = ndimage.gaussian_filter(diff_vis, sigma=args.sigma)
    
    # Apply colormap
    cmap = plt.get_cmap(args.colormap)
    heatmap_1_colored = cmap(heatmap_1_vis)[:, :, :3]
    heatmap_2_colored = cmap(heatmap_2_vis)[:, :, :3]
    
    # Create overlays
    overlay_1 = (1 - args.alpha) * img_array + args.alpha * heatmap_1_colored
    overlay_1 = np.clip(overlay_1 * 255, 0, 255).astype(np.uint8)
    overlay_2 = (1 - args.alpha) * img_array + args.alpha * heatmap_2_colored
    overlay_2 = np.clip(overlay_2 * 255, 0, 255).astype(np.uint8)
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Panel 1: Original
    axes[0].imshow(img_array)
    axes[0].set_title(f"Original Image\n{unique_id}", fontsize=10)
    axes[0].axis("off")
    
    # Panel 2: Model 1
    axes[1].imshow(overlay_1)
    axes[1].set_title(f"{model_name_1}\npred={pred_1} | p(class1)={prob_1:.3f}", fontsize=10)
    axes[1].axis("off")
    
    # Panel 3: Model 2
    axes[2].imshow(overlay_2)
    axes[2].set_title(f"{model_name_2}\npred={pred_2} | p(class1)={prob_2:.3f}", fontsize=10)
    axes[2].axis("off")
    
    # Panel 4: Difference
    im = axes[3].imshow(img_array)
    diff_cmap = args.diff_cmap if hasattr(args, 'diff_cmap') else "coolwarm"
    im2 = axes[3].imshow(diff_vis, cmap=diff_cmap, alpha=args.alpha, interpolation='bilinear', vmin=-1, vmax=1)
    axes[3].set_title(f"Δ Attention ({model_name_2} − {model_name_1})", fontsize=10)
    axes[3].axis("off")
    plt.colorbar(im2, ax=axes[3], label="Δ Attention", fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    


def normalize_for_vis_comparison(
    heatmap: np.ndarray,
    clip_low: float = 90.0,
    clip_high: float = 99.5,
    gamma: float = 0.7
) -> np.ndarray:
    """Normalize heatmap for visualization (comparison mode)."""
    heatmap_flat = heatmap.flatten()
    valid_mask = np.isfinite(heatmap_flat) & (heatmap_flat > 0)
    
    if valid_mask.sum() == 0:
        return np.zeros_like(heatmap)
    
    valid_values = heatmap_flat[valid_mask]
    low_val = np.percentile(valid_values, clip_low)
    high_val = np.percentile(valid_values, clip_high)
    
    if high_val <= low_val:
        high_val = valid_values.max()
        low_val = valid_values.min()
    
    heatmap_clipped = np.clip(heatmap, low_val, high_val)
    heatmap_norm = (heatmap_clipped - low_val) / (high_val - low_val + 1e-8)
    heatmap_norm = np.power(heatmap_norm, gamma)
    
    return heatmap_norm


# ---------------------------------------------------------------------------
# Lightweight Attention Capture
# ---------------------------------------------------------------------------

def install_lightweight_attention_capture(vision_model, cls_index, n_special, capture_last_n=1, captured_attentions_list=None):
    """
    Install lightweight monkey-patch to capture only CLS-to-patch attention rows.
    Deterministically patches only transformer.layers[i].self_attn modules (NOT submodules).
    
    Args:
        vision_model: The vision encoder model
        cls_index: Index of CLS token
        n_special: Number of special tokens (patches start after this)
        capture_last_n: Number of last layers to capture (1=last only, N=last N for rollout)
        captured_attentions_list: List to store captured attention rows [B, H, patch_tokens]
    
    Returns:
        original_forwards: Dict of original forward methods for cleanup
    """
    # Use provided list or create new one
    if captured_attentions_list is None:
        captured_attentions_list = []
    
    original_forwards = {}
    
    # Deterministically find transformer layers (NO broad scanning)
    if not hasattr(vision_model, 'transformer') or not hasattr(vision_model.transformer, 'layers'):
        raise ValueError(
            f"Expected vision_model.transformer.layers structure. "
            f"Found: {list(vision_model._modules.keys())}"
        )
    
    layers = vision_model.transformer.layers
    num_layers = len(layers)
    
    # Determine which layers to capture
    capture_indices = list(range(max(0, num_layers - capture_last_n), num_layers))
    
    def create_lightweight_wrapper(original_forward, layer_idx, attn_list, layer_module):
        """Wrapper that computes attention and captures only CLS-to-patch row (lightweight!)"""
        def wrapped_forward(hidden_states, attention_mask=None, **kwargs):
            # Call original forward to get output (model still uses original behavior)
            result = original_forward(hidden_states, attention_mask=attention_mask, **kwargs)
            
            # Extract attn_output (model still uses original behavior)
            if isinstance(result, tuple):
                attn_output = result[0]
            else:
                attn_output = result
            
            # Only capture if this layer is in capture_indices
            if layer_idx in capture_indices:
                try:
                    # Get module attributes
                    module = layer_module.self_attn
                    num_heads = getattr(module, 'num_heads', 
                                       getattr(module, 'num_attention_heads', None))
                    if num_heads is None and hasattr(module, 'config'):
                        num_heads = getattr(module.config, 'num_attention_heads', None)
                    
                    embed_dim = hidden_states.shape[-1]
                    if num_heads is None:
                        head_dim = 64  # Common for ViT
                        num_heads = embed_dim // head_dim
                    
                    head_dim = embed_dim // num_heads
                    
                    # Get projections
                    q_proj = getattr(module, 'q_proj', None)
                    k_proj = getattr(module, 'k_proj', None)
                    
                    if q_proj is None or k_proj is None:
                        return result  # Skip capture if can't compute
                    
                    batch_size, seq_len = hidden_states.shape[:2]
                    
                    # Project to Q, K (we don't need V for attention weights)
                    query_states = q_proj(hidden_states)
                    key_states = k_proj(hidden_states)
                    
                    # Reshape: [B, T, H*d] -> [B, H, T, d]
                    query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                    key_states = key_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                    
                    # Compute attention scores: [B, H, T, T]
                    attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1))
                    scale_factor = 1.0 / (head_dim ** 0.5)
                    attn_scores = attn_scores * scale_factor
                    
                    # Apply mask if provided
                    if attention_mask is not None:
                        if attention_mask.dim() == 2:
                            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                        elif attention_mask.dim() == 3:
                            attention_mask = attention_mask.unsqueeze(1)
                        
                        if attention_mask.dtype == torch.bool:
                            attn_mask_expanded = attention_mask.expand(batch_size, num_heads, seq_len, seq_len)
                            attn_scores = attn_scores.masked_fill(~attn_mask_expanded, float('-inf'))
                        else:
                            attn_scores = attn_scores + attention_mask
                    
                    # Compute attention probabilities
                    attn_probs = F.softmax(attn_scores, dim=-1)
                    
                    # Extract CLS-to-patch row ONLY: [B, H, patch_tokens] - LIGHTWEIGHT!
                    cls_attention_row = attn_probs[:, :, cls_index, :]  # [B, H, T]
                    cls_to_patches = cls_attention_row[:, :, n_special:]  # [B, H, patch_tokens]
                    
                    # Store as float32 on CPU (lightweight!)
                    attn_list.append(cls_to_patches.detach().float().cpu())
                    
                except Exception as e:
                    pass
            
            return result
        
        return wrapped_forward
    
    # Patch only the layers we need (deterministic path)
    for i in capture_indices:
        layer = layers[i]
        if not hasattr(layer, 'self_attn'):
            raise ValueError(f"Layer {i} does not have self_attn attribute")
        
        attn_module = layer.self_attn
        layer_name = f"transformer.layers.{i}.self_attn"
        
        if layer_name not in original_forwards:
            original_forwards[layer_name] = attn_module.forward
            attn_module.forward = create_lightweight_wrapper(attn_module.forward, i, captured_attentions_list, layer)
    
    return original_forwards


# ---------------------------------------------------------------------------
# Single Model Processing
# ---------------------------------------------------------------------------

def process_single_model(
    model: nn.Module,
    processor: AutoProcessor,
    sample: Dict,
    args,
    device: torch.device,
    checkpoint_dir: str,
    model_name: str = "Model"
) -> Dict:
    """Process a single model and return attention heatmap and predictions."""
    model = model.to(device)
    model.eval()
    
    # Force eager attention
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    
    if hasattr(model.base_model.vision_model.config, 'attn_implementation'):
        model.base_model.vision_model.config.attn_implementation = "eager"
    if hasattr(model.config, 'attn_implementation'):
        model.config.attn_implementation = "eager"
    
    # Prepare input
    system_prompt = (
        "A clinical document and a single, most recent chest X-ray (CXR) image from the patient are provided. "
        "Based on the clinical context and the provided CXR image, assess how likely the patient's out-of-hospital mortality is within 30 days."
    )
    
    personal_information = ""
    if sample.get('discharge_note'):
        user_prompt = (
            f"Here is the clinical document:\n{personal_information} {sample['discharge_note']}\n\n"
            "Based on the provided clinical information and single CXR image, how likely is the patient's out-of-hospital mortality within 30 days? "
            "Please do not explain the reason and respond with one word only: 0:alive, 1:death.\n\nAssistant:"
        )
    else:
        user_prompt = (
            "Based on the provided single CXR image, how likely is the patient's out-of-hospital mortality within 30 days? "
            "Please do not explain the reason and respond with one word only: 0:alive, 1:death.\n\nAssistant:"
        )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_prompt}
        ]}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False)
    processed = processor(
        text=[text],
        images=[[sample['image']]],
        return_tensors="pt",
        padding=True
    )
    
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in processed.items()}
    
    # ========================================================================
    # CRITICAL: Capture attention BEFORE full model forward to ensure
    # vision encoder is not affected by projector/LM state
    # ========================================================================
    pixel_values = batch['pixel_values']
    aspect_ratio_ids = batch.get('aspect_ratio_ids', None)
    aspect_ratio_mask = batch.get('aspect_ratio_mask', None)
    
    vision_model = model.base_model.vision_model
    
    # Detect tokens and grid
    cls_index, n_special, num_patches, grid_h, grid_w = detect_vision_tokens_and_grid(
        vision_model, pixel_values, processor, aspect_ratio_ids, aspect_ratio_mask
    )
    
    # Install lightweight capture
    captured_attentions = []
    original_forwards = {}
    
    capture_last_n = 1 if args.mode == "last" else args.rollout_layers
    original_forwards = install_lightweight_attention_capture(
        vision_model, cls_index, n_special, capture_last_n, captured_attentions
    )
    
    # Run vision encoder forward pass to capture attention (ISOLATED)
    with torch.no_grad():
        vision_kwargs = {"pixel_values": pixel_values}
        if aspect_ratio_ids is not None:
            vision_kwargs["aspect_ratio_ids"] = aspect_ratio_ids
        if aspect_ratio_mask is not None:
            vision_kwargs["aspect_ratio_mask"] = aspect_ratio_mask
        _ = vision_model(**vision_kwargs)
    
    # ========================================================================
    # NOW run full model forward for predictions (after attention captured)
    # ========================================================================
    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs["logits"]
        probabilities = F.softmax(logits.float(), dim=-1)
    
    prob_alive = float(probabilities[0][0].item())
    prob_death = float(probabilities[0][1].item())
    prediction = int(torch.argmax(probabilities[0]).item())
    
    # Verify capture
    if not captured_attentions:
        raise RuntimeError("No attentions captured!")
    
    first_attn = captured_attentions[0]
    if len(first_attn.shape) != 3:
        raise ValueError(f"Expected [B, H, patch_tokens], got shape {first_attn.shape}")
    
    actual_patch_tokens = first_attn.shape[2]
    
    # Get vision_meta
    vision_meta = None
    try:
        with torch.no_grad():
            if len(pixel_values.shape) == 6:
                pv_test = pixel_values[:1, :, :, :, :, :]
            else:
                pv_test = pixel_values[:1]
            test_vision_out = vision_model(pv_test, return_dict=True, 
                                         aspect_ratio_ids=aspect_ratio_ids[:1] if aspect_ratio_ids is not None and aspect_ratio_ids.dim() > 0 else aspect_ratio_ids,
                                         aspect_ratio_mask=aspect_ratio_mask[:1] if aspect_ratio_mask is not None and aspect_ratio_mask.dim() > 0 else aspect_ratio_mask)
            if hasattr(test_vision_out, 'last_hidden_state'):
                _, vision_meta = normalize_vision_hidden_state(test_vision_out.last_hidden_state)
    except:
        pass
    
    # Handle tiled mode
    Ntiles = vision_meta['Ntiles'] if vision_meta and vision_meta['Ntiles'] > 1 else 1
    per_tile_patches_original = num_patches
    per_tile_grid_h_original = grid_h
    per_tile_grid_w_original = grid_w
    
    if vision_meta and vision_meta['Ntiles'] > 1:
        per_tile_expected = num_patches
        total_expected = per_tile_expected * vision_meta['Ntiles']
        
        if actual_patch_tokens >= total_expected:
            per_tile_patches_original = num_patches
            per_tile_grid_h_original = grid_h
            per_tile_grid_w_original = grid_w
    
    # Compute heatmap
    heatmaps_per_tile = None
    valid_tiles = list(range(Ntiles))
    
    if Ntiles > 1:
        if aspect_ratio_mask is not None:
            mask_flat = aspect_ratio_mask.flatten()
            if len(mask_flat) >= Ntiles:
                valid_tiles = []
                for tile_idx in range(Ntiles):
                    if tile_idx < len(mask_flat):
                        is_valid = mask_flat[tile_idx].item() > 0 if mask_flat.dtype != torch.bool else mask_flat[tile_idx].item()
                        if is_valid:
                            valid_tiles.append(tile_idx)
                
                if not valid_tiles:
                    valid_tiles = list(range(Ntiles))
        
        per_tile_T = vision_meta['T']
        per_tile_patches = per_tile_patches_original
        
        heatmaps_per_tile = []
        for tile_idx in range(Ntiles):
            patch_start_in_captured = tile_idx * per_tile_patches
            patch_end_in_captured = (tile_idx + 1) * per_tile_patches
            
            tile_attentions = []
            for attn in captured_attentions:
                if attn.shape[2] >= patch_end_in_captured:
                    tile_patch_attn = attn[:, :, patch_start_in_captured:patch_end_in_captured]
                    # Ensure it's on CPU and detach
                    if isinstance(tile_patch_attn, torch.Tensor):
                        tile_patch_attn = tile_patch_attn.detach().cpu()
                    tile_attentions.append(tile_patch_attn)
                else:
                    tile_patch_attn = torch.zeros(1, attn.shape[1], per_tile_patches, dtype=attn.dtype)
                    tile_attentions.append(tile_patch_attn)
            
            per_tile_grid_h = per_tile_grid_h_original
            per_tile_grid_w = per_tile_grid_w_original
            
            if args.mode == "last":
                tile_heatmap = compute_last_block_attention(tile_attentions, per_tile_grid_h, per_tile_grid_w)
            else:
                tile_heatmap = compute_attention_rollout(tile_attentions, per_tile_grid_h, per_tile_grid_w)
            
            # Convert to numpy immediately to free GPU memory
            if isinstance(tile_heatmap, torch.Tensor):
                tile_heatmap = tile_heatmap.cpu().numpy()
            elif not isinstance(tile_heatmap, np.ndarray):
                tile_heatmap = np.array(tile_heatmap)
            
            heatmaps_per_tile.append(tile_heatmap)
            
            # Clean up tile_attentions immediately
            del tile_attentions, tile_patch_attn
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Select representative tile
        tile_metrics = []
        for tile_idx in valid_tiles:
            if tile_idx < len(heatmaps_per_tile):
                tile_heatmap = heatmaps_per_tile[tile_idx]
                heatmap_flat = tile_heatmap.flatten()
                valid_mask = np.isfinite(heatmap_flat) & (heatmap_flat > 0)
                
                if valid_mask.sum() == 0:
                    continue
                
                valid_values = heatmap_flat[valid_mask]
                valid_sum = valid_values.sum()
                if valid_sum > 0:
                    probs = valid_values / valid_sum
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    top_5_percent_idx = int(len(valid_values) * 0.95)
                    top_5_percent_mass = np.sort(valid_values)[top_5_percent_idx:].sum() / valid_sum if valid_sum > 0 else 0.0
                    
                    tile_metrics.append({
                        'tile_idx': tile_idx,
                        'entropy': entropy,
                        'top_5_percent_mass': top_5_percent_mass
                    })
        
        if tile_metrics:
            tile_metrics.sort(key=lambda x: (-x['top_5_percent_mass'], x['entropy']))
            representative_tile = tile_metrics[0]['tile_idx']
            
            # Combine valid tiles for main overlay (sum all valid tiles)
            if len(valid_tiles) > 1:
                combined_heatmap = np.zeros_like(heatmaps_per_tile[valid_tiles[0]])
                for tile_idx in valid_tiles:
                    if tile_idx < len(heatmaps_per_tile):
                        combined_heatmap += heatmaps_per_tile[tile_idx]
                heatmap = combined_heatmap
            else:
                # Use representative tile for main overlay (single valid tile)
                heatmap = heatmaps_per_tile[representative_tile]
        else:
            representative_tile = valid_tiles[0] if valid_tiles else 0
            # Combine valid tiles if multiple, otherwise use single tile
            if len(valid_tiles) > 1:
                combined_heatmap = np.zeros_like(heatmaps_per_tile[valid_tiles[0]])
                for tile_idx in valid_tiles:
                    if tile_idx < len(heatmaps_per_tile):
                        combined_heatmap += heatmaps_per_tile[tile_idx]
                heatmap = combined_heatmap
            else:
                heatmap = heatmaps_per_tile[representative_tile]
    else:
        if args.mode == "last":
            heatmap = compute_last_block_attention(captured_attentions, grid_h, grid_w)
        else:
            heatmap = compute_attention_rollout(captured_attentions, grid_h, grid_w)
        representative_tile = 0
    
    # Restore original forwards
    for layer_name, original_forward in original_forwards.items():
        parts = layer_name.split('.')
        module = vision_model
        for part in parts:
            module = getattr(module, part)
        # Now module is the self_attn module, restore its forward method
        module.forward = original_forward
    
    # Clean up GPU memory
    del batch, outputs, logits, probabilities
    del pixel_values, aspect_ratio_ids, aspect_ratio_mask
    if 'test_vision_out' in locals():
        del test_vision_out
    if 'tile_attentions' in locals():
        del tile_attentions
    if 'tile_heatmap' in locals():
        del tile_heatmap
    
    # Clear captured attentions from GPU (they should already be on CPU, but double-check)
    for attn in captured_attentions:
        if isinstance(attn, torch.Tensor) and attn.is_cuda:
            attn = attn.cpu()
    captured_attentions.clear()
    
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Convert heatmap to numpy if it's still a tensor
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    elif not isinstance(heatmap, np.ndarray):
        heatmap = np.array(heatmap)
    
    # Convert heatmaps_per_tile to numpy if needed
    if heatmaps_per_tile is not None:
        heatmaps_per_tile_np = []
        for tile_hm in heatmaps_per_tile:
            if isinstance(tile_hm, torch.Tensor):
                heatmaps_per_tile_np.append(tile_hm.cpu().numpy())
            elif not isinstance(tile_hm, np.ndarray):
                heatmaps_per_tile_np.append(np.array(tile_hm))
            else:
                heatmaps_per_tile_np.append(tile_hm)
        heatmaps_per_tile = heatmaps_per_tile_np
    
    # ========================================================================
    # Compute gradient attribution if requested
    # ========================================================================
    gradient_heatmap = None
    if args.do_gradient:
        try:
            # Recreate batch for gradient computation (fresh copy)
            text = processor.apply_chat_template(messages, tokenize=False)
            processed = processor(
                text=[text],
                images=[[sample['image']]],
                return_tensors="pt",
                padding=True
            )
            batch_grad = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in processed.items()}
            
            gradient_heatmap = compute_gradient_attribution(
                model, batch_grad, device,
                target_class=1,
                grid_h=grid_h,
                grid_w=grid_w,
                cls_index=cls_index,
                n_special=n_special,
                num_patches=num_patches,
                representative_tile=representative_tile,
                vision_meta=vision_meta
            )
            
            del batch_grad, processed
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception:
            gradient_heatmap = None
    
    # ========================================================================
    # Compute occlusion sensitivity if requested
    # ========================================================================
    occlusion_heatmap = None
    base_death_logit = None
    if args.do_occlusion:
        try:
            # Recreate batch for occlusion (fresh copy)
            text = processor.apply_chat_template(messages, tokenize=False)
            processed = processor(
                text=[text],
                images=[[sample['image']]],
                return_tensors="pt",
                padding=True
            )
            batch_occ = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in processed.items()}
            
            occlusion_heatmap, base_death_logit = compute_occlusion_sensitivity(
                model, batch_occ, device, processor, sample,
                target_class=1,
                grid_h=grid_h,
                grid_w=grid_w,
                representative_tile=representative_tile,
                vision_meta=vision_meta,
                batch_size=args.occlusion_batch_size,
                mask_value=args.occlusion_mask_value,
                occlusion_mode=args.occlusion_mode
            )
            
            del batch_occ, processed
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception:
            occlusion_heatmap = None
            base_death_logit = None
    
    return {
        "checkpoint_dir": checkpoint_dir,
        "prediction": prediction,
        "probability": {
            "class_0": prob_alive,
            "class_1": prob_death
        },
        "n_tiles": Ntiles,
        "valid_tiles": valid_tiles,
        "representative_tile": representative_tile,
        "heatmap": heatmap,
        "heatmaps_per_tile": heatmaps_per_tile if heatmaps_per_tile else None,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "vision_meta": vision_meta,
        "gradient_heatmap": gradient_heatmap,
        "occlusion_heatmap": occlusion_heatmap,
        "base_death_logit": base_death_logit
    }


# ---------------------------------------------------------------------------
# Main Function
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize ViT attention from LLaMA 3.2 Vision encoder with unique_id"
    )
    parser.add_argument(
        "--unique_id",
        type=str,
        required=True,
        help="Unique ID of the sample to visualize"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./attention_outputs",
        help="Output directory for visualizations and JSON"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="last",
        choices=["last", "rollout"],
        help="Attention visualization mode"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.2-11B-Vision-Instruct",
        help="Hugging Face model identifier"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="../trained_models/dn",
        help="Checkpoint directory (for single model mode)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Enable comparison mode (requires checkpoint_dir_1 and checkpoint_dir_2)"
    )
    parser.add_argument(
        "--checkpoint_dir_1",
        type=str,
        default=None,
        help="First checkpoint directory (for comparison mode)"
    )
    parser.add_argument(
        "--checkpoint_dir_2",
        type=str,
        default=None,
        help="Second checkpoint directory (for comparison mode)"
    )
    parser.add_argument(
        "--model_name_1",
        type=str,
        default="Model 1",
        help="Name for first model in comparison"
    )
    parser.add_argument(
        "--model_name_2",
        type=str,
        default="Model 2",
        help="Name for second model in comparison"
    )
    parser.add_argument(
        "--diff_clip",
        type=float,
        default=99.0,
        help="Percentile for difference map clipping (default: 99.0)"
    )
    parser.add_argument(
        "--diff_cmap",
        type=str,
        default="coolwarm",
        choices=["coolwarm", "RdBu", "seismic", "bwr"],
        help="Colormap for difference map (default: coolwarm)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Overlay transparency (default: 0.35)"
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="viridis",
        choices=["magma", "viridis", "plasma", "inferno", "jet"],
        help="Matplotlib colormap (default: magma)"
    )
    parser.add_argument(
        "--rollout_layers",
        type=int,
        default=6,
        help="Number of last layers to use for rollout (if mode=rollout, default: 6)"
    )
    parser.add_argument(
        "--clip_low",
        type=float,
        default=5.0,
        help="Lower percentile for clipping (default: 5.0)"
    )
    parser.add_argument(
        "--clip_high",
        type=float,
        default=99.0,
        help="Upper percentile for clipping (default: 99.0)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        help="Gamma correction (default: 0.7)"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=5.0,
        help="Gaussian smoothing sigma in pixels (default: 5.0, set 0 to disable)"
    )
    parser.add_argument(
        "--border_mask",
        type=float,
        default=0.05,
        help="Border mask ratio (default: 0.05, set 0 to disable)"
    )
    
    # Gradient attribution and occlusion
    parser.add_argument(
        "--do_gradient",
        action="store_true",
        default=False,
        help="Compute gradient-based attribution (default: False)"
    )
    parser.add_argument(
        "--do_occlusion",
        action="store_true",
        default=False,
        help="Compute occlusion sensitivity (default: False)"
    )
    parser.add_argument(
        "--occlusion_batch_size",
        type=int,
        default=128,
        help="Batch size for occlusion computation (default: 8)"
    )
    parser.add_argument(
        "--occlusion_mask_value",
        type=float,
        default=0.0,
        help="Mask value for occlusion in normalized pixel space (default: 0.0)"
    )
    parser.add_argument(
        "--occlusion_mode",
        type=str,
        default="patch",
        choices=["patch", "blur"],
        help="Occlusion mode: patch masking or blur (default: patch)"
    )
    
    # Data paths (with defaults matching codebase structure)
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="../dataset/metadata.json"
    )
    parser.add_argument(
        "--metadata_image_path",
        type=str,
        default="../dataset/test_summarization/full-test-indent-images.json"
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="../dataset/test_summarization/total_output.jsonl"
    )
    parser.add_argument(
        "--base_img_dir",
        type=str,
        default="../saved_images"
    )
    parser.add_argument(
        "--base_rr_dir",
        type=str,
        default="../physionet.org/files/mimic-cxr/2.1.0/files"
    )
    parser.add_argument(
        "--summary_type",
        type=str,
        default="plain"
    )
    
    args = parser.parse_args()
    
    # Validate comparison mode arguments
    if args.compare:
        if args.checkpoint_dir_1 is None or args.checkpoint_dir_2 is None:
            raise ValueError("Comparison mode requires both --checkpoint_dir_1 and --checkpoint_dir_2")
    
    # Setup device
    device = torch.device(args.device)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Create args object for model loading
    from dataclasses import dataclass
    
    @dataclass
    class ModelArgs:
        model_name_or_path: str = args.model_id
        checkpoint_dir: str = args.checkpoint_dir
        use_cxr_image: bool = True
        use_discharge_note: bool = True
        use_rad_report: bool = False
        use_generated_rad_report: bool = False
        use_pi: bool = False
        summarize: bool = False
        zeroshot: bool = False
        summary_type: str = args.summary_type
        base_img_dir: str = args.base_img_dir
        base_rr_dir: str = args.base_rr_dir
        metadata_path: str = args.metadata_path
        test_data_path: str = args.test_data_path
        test_metadata_image_path: str = args.metadata_image_path
    
    # Load sample (needed for both modes)
    from utils import load_adapter, map_adapter_keys
    processor = AutoProcessor.from_pretrained(args.model_id)
    sample = load_sample_by_unique_id(args.unique_id, args, processor)
    
    # ========================================================================
    # COMPARISON MODE
    # ========================================================================
    if args.compare:
        
        # Process Model 1
        model_args_1 = ModelArgs()
        model_args_1.checkpoint_dir = args.checkpoint_dir_1
        model_args_1.inference = True
        
        model_1, processor_1 = load_model(model_args_1, model_id=args.model_id, inference=True)
        result_1 = process_single_model(model_1, processor_1, sample, args, device, args.checkpoint_dir_1, args.model_name_1)
        
        del model_1, processor_1
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Process Model 2
        model_args_2 = ModelArgs()
        model_args_2.checkpoint_dir = args.checkpoint_dir_2
        model_args_2.inference = True
        
        model_2, processor_2 = load_model(model_args_2, model_id=args.model_id, inference=True)
        result_2 = process_single_model(model_2, processor_2, sample, args, device, args.checkpoint_dir_2, args.model_name_2)
        
        del model_2, processor_2
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Compute difference map (ensure both are numpy arrays)
        heatmap_1 = np.array(result_1["heatmap"], dtype=np.float32)
        heatmap_2 = np.array(result_2["heatmap"], dtype=np.float32)
        diff_map = heatmap_2 - heatmap_1
        
        # Create comparison figure (attention only for comparison mode)
        output_base = Path(args.out_dir)
        output_base.mkdir(parents=True, exist_ok=True)
        output_path = output_base / f"{args.unique_id}_comparison.png"
        create_comparison_figure(
            sample['image'],
            heatmap_1,
            heatmap_2,
            diff_map,
            result_1["prediction"],
            result_1["probability"]["class_1"],
            result_2["prediction"],
            result_2["probability"]["class_1"],
            result_1["valid_tiles"],
            result_2["valid_tiles"],
            result_1["representative_tile"],
            result_2["representative_tile"],
            output_path,
            args.unique_id,
            args.model_name_1,
            args.model_name_2,
            args
        )
        
        # Save summary JSON
        summary = {
            "unique_id": args.unique_id,
            "discharge_note": sample.get('discharge_note'),
            "radiology_report": sample.get('radiology_report'),
            "generated_radiology_report": sample.get('generated_radiology_report'),
            "image_path": sample.get('image_path'),
            "model_1": {
                "checkpoint_dir": result_1["checkpoint_dir"],
                "name": args.model_name_1,
                "prediction": result_1["prediction"],
                "prob_class_1": result_1["probability"]["class_1"],
                "n_tiles": result_1["n_tiles"],
                "valid_tiles": result_1["valid_tiles"],
                "representative_tile": result_1["representative_tile"]
            },
            "model_2": {
                "checkpoint_dir": result_2["checkpoint_dir"],
                "name": args.model_name_2,
                "prediction": result_2["prediction"],
                "prob_class_1": result_2["probability"]["class_1"],
                "n_tiles": result_2["n_tiles"],
                "valid_tiles": result_2["valid_tiles"],
                "representative_tile": result_2["representative_tile"]
            },
            "attention_mode": args.mode,
            "visualization_params": {
                "alpha": args.alpha,
                "colormap": args.colormap,
                "clip_percentiles": [args.clip_low, args.clip_high],
                "gamma": args.gamma,
                "sigma": args.sigma,
                "border_mask_ratio": args.border_mask,
                "diff_clip": args.diff_clip,
                "diff_cmap": args.diff_cmap
            }
        }
        
        summary_path = output_base / f"{args.unique_id}_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return
    
    # ========================================================================
    # SINGLE MODEL MODE (Original behavior)
    # ========================================================================
    
    model_args = ModelArgs()
    model_args.inference = True  # Mark as inference mode
    
    # Load model
    model, processor = load_model(model_args, model_id=args.model_id, inference=True)
    model = model.to(device)
    model.eval()
    
    # ========================================================================
    # BLOCK A: Force Eager Attention Implementation
    # ========================================================================
    # Disable Flash Attention and Memory-Efficient SDPA
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    
    # Force eager attention implementation in configs
    if hasattr(model.base_model.vision_model.config, 'attn_implementation'):
        original_attn_impl = model.base_model.vision_model.config.attn_implementation
        model.base_model.vision_model.config.attn_implementation = "eager"
    
    if hasattr(model.config, 'attn_implementation'):
        original_attn_impl = model.config.attn_implementation
        model.config.attn_implementation = "eager"
    
    # Also try to set on vision_model directly if it has the attribute
    if hasattr(model.base_model.vision_model, 'config'):
        if hasattr(model.base_model.vision_model.config, 'attn_implementation'):
            model.base_model.vision_model.config.attn_implementation = "eager"
    
    # ========================================================================
    # B) Lightweight Attention Capture (CLS-to-patch only)
    # ========================================================================
    
    # Global storage for captured attentions (CLS-to-patch rows only, float32 on CPU)
    captured_attentions = []
    original_forwards = {}
    
    def install_lightweight_attention_capture(vision_model, cls_index, n_special, capture_last_n=1, captured_attentions_list=None):
        """
        Install lightweight monkey-patch to capture only CLS-to-patch attention rows.
        Deterministically patches only transformer.layers[i].self_attn modules (NOT submodules).
        
        Args:
            vision_model: The vision encoder model
            cls_index: Index of CLS token
            n_special: Number of special tokens (patches start after this)
            capture_last_n: Number of last layers to capture (1=last only, N=last N for rollout)
            captured_attentions_list: List to store captured attention rows [B, H, patch_tokens]
        
        Returns:
            original_forwards: Dict of original forward methods for cleanup
        """
        # Use provided list or global one
        if captured_attentions_list is None:
            captured_attentions_list = captured_attentions
        
        # Deterministically find transformer layers (NO broad scanning)
        if not hasattr(vision_model, 'transformer') or not hasattr(vision_model.transformer, 'layers'):
            raise ValueError(
                f"Expected vision_model.transformer.layers structure. "
                f"Found: {list(vision_model._modules.keys())}"
            )
        
        layers = vision_model.transformer.layers
        num_layers = len(layers)
        
        # Determine which layers to capture
        capture_indices = list(range(max(0, num_layers - capture_last_n), num_layers))
        
        def create_lightweight_wrapper(original_forward, layer_idx, attn_list):
            """Wrapper that computes attention and captures only CLS-to-patch row (lightweight!)"""
            def wrapped_forward(hidden_states, attention_mask=None, **kwargs):
                # Call original forward to get output (model still uses original behavior)
                result = original_forward(hidden_states, attention_mask=attention_mask, **kwargs)
                
                # Extract attn_output (model still uses original behavior)
                if isinstance(result, tuple):
                    attn_output = result[0]
                else:
                    attn_output = result
                
                # Only capture if this layer is in capture_indices
                if layer_idx in capture_indices:
                    try:
                        # Get module attributes
                        module = layers[layer_idx].self_attn
                        num_heads = getattr(module, 'num_heads', 
                                           getattr(module, 'num_attention_heads', None))
                        if num_heads is None and hasattr(module, 'config'):
                            num_heads = getattr(module.config, 'num_attention_heads', None)
                        
                        embed_dim = hidden_states.shape[-1]
                        if num_heads is None:
                            head_dim = 64  # Common for ViT
                            num_heads = embed_dim // head_dim
                        
                        head_dim = embed_dim // num_heads
                        
                        # Get projections
                        q_proj = getattr(module, 'q_proj', None)
                        k_proj = getattr(module, 'k_proj', None)
                        
                        if q_proj is None or k_proj is None:
                            return result  # Skip capture if can't compute
                        
                        batch_size, seq_len = hidden_states.shape[:2]
                        
                        # Project to Q, K (we don't need V for attention weights)
                        query_states = q_proj(hidden_states)
                        key_states = k_proj(hidden_states)
                        
                        # Reshape: [B, T, H*d] -> [B, H, T, d]
                        query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                        key_states = key_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
                        
                        # Compute attention scores: [B, H, T, T]
                        attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1))
                        scale_factor = 1.0 / (head_dim ** 0.5)
                        attn_scores = attn_scores * scale_factor
                        
                        # Apply mask if provided (store original for debug)
                        attn_mask_original = attention_mask
                        if attention_mask is not None:
                            if attention_mask.dim() == 2:
                                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                            elif attention_mask.dim() == 3:
                                attention_mask = attention_mask.unsqueeze(1)
                            
                            if attention_mask.dtype == torch.bool:
                                attn_mask_expanded = attention_mask.expand(batch_size, num_heads, seq_len, seq_len)
                                attn_scores = attn_scores.masked_fill(~attn_mask_expanded, float('-inf'))
                            else:
                                attn_scores = attn_scores + attention_mask
                        
                        # Compute attention probabilities
                        attn_probs = F.softmax(attn_scores, dim=-1)
                        
                        # Extract CLS-to-patch row ONLY: [B, H, patch_tokens] - LIGHTWEIGHT!
                        cls_attention_row = attn_probs[:, :, cls_index, :]  # [B, H, T]
                        cls_to_patches = cls_attention_row[:, :, n_special:]  # [B, H, patch_tokens]
                        
                        # Store as float32 on CPU (lightweight!)
                        attn_list.append(cls_to_patches.detach().float().cpu())
                        
                    except Exception as e:
                        pass
                
                return result
            
            return wrapped_forward
        
        # Patch only the layers we need (deterministic path)
        for i in capture_indices:
            layer = layers[i]
            if not hasattr(layer, 'self_attn'):
                raise ValueError(f"Layer {i} does not have self_attn attribute")
            
            attn_module = layer.self_attn
            layer_name = f"transformer.layers.{i}.self_attn"
            
            if layer_name not in original_forwards:
                original_forwards[layer_name] = attn_module.forward
                attn_module.forward = create_lightweight_wrapper(attn_module.forward, i, captured_attentions_list)
    
        return original_forwards
    
    # Load sample by unique_id (needed for dummy forward to detect tokens)
    sample = load_sample_by_unique_id(args.unique_id, args, processor)
    
    # Prepare input for model
    # Create prompt with discharge note + image
    system_prompt = (
        "A clinical document and a single, most recent chest X-ray (CXR) image from the patient are provided. "
        "Based on the clinical context and the provided CXR image, assess how likely the patient's out-of-hospital mortality is within 30 days."
    )
    
    personal_information = ""
    if sample.get('discharge_note'):
        user_prompt = (
            f"Here is the clinical document:\n{personal_information} {sample['discharge_note']}\n\n"
            "Based on the provided clinical information and single CXR image, how likely is the patient's out-of-hospital mortality within 30 days? "
            "Please do not explain the reason and respond with one word only: 0:alive, 1:death.\n\nAssistant:"
        )
    else:
        user_prompt = (
            "Based on the provided single CXR image, how likely is the patient's out-of-hospital mortality within 30 days? "
            "Please do not explain the reason and respond with one word only: 0:alive, 1:death.\n\nAssistant:"
        )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_prompt}
        ]}
    ]
    
    # Process with processor
    text = processor.apply_chat_template(messages, tokenize=False)
    processed = processor(
        text=[text],
        images=[[sample['image']]],
        return_tensors="pt",
        padding=True
    )
    
    # Move to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in processed.items()}
    
    # Run inference
    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs["logits"]
        probabilities = F.softmax(logits.float(), dim=-1)
    
    prob_alive = float(probabilities[0][0].item())
    prob_death = float(probabilities[0][1].item())
    prediction = int(torch.argmax(probabilities[0]).item())
    
    
    # Detect tokens and grid BEFORE installing patch
    pixel_values = batch['pixel_values']
    aspect_ratio_ids = batch.get('aspect_ratio_ids', None)
    aspect_ratio_mask = batch.get('aspect_ratio_mask', None)
    vision_model = model.base_model.vision_model
    
    cls_index, n_special, num_patches, grid_h, grid_w = detect_vision_tokens_and_grid(
        vision_model, pixel_values, processor, aspect_ratio_ids, aspect_ratio_mask
    )
    
    # Install lightweight capture (deterministic, only patches transformer.layers[i].self_attn)
    capture_last_n = 1 if args.mode == "last" else args.rollout_layers
    original_forwards = install_lightweight_attention_capture(
        vision_model, cls_index, n_special, capture_last_n, captured_attentions
    )
    
    # Run forward pass to capture (lightweight: only CLS-to-patch rows)
    with torch.no_grad():
        vision_kwargs = {"pixel_values": pixel_values}
        if aspect_ratio_ids is not None:
            vision_kwargs["aspect_ratio_ids"] = aspect_ratio_ids
        if aspect_ratio_mask is not None:
            vision_kwargs["aspect_ratio_mask"] = aspect_ratio_mask
        _ = vision_model(**vision_kwargs)
    
    # Verify capture and update num_patches/grid from actual captured tensors
    if not captured_attentions:
        raise RuntimeError("No attentions captured!")
    
    # Get actual patch token count from first captured tensor
    first_attn = captured_attentions[0]
    if len(first_attn.shape) != 3:
        raise ValueError(f"Expected [B, H, patch_tokens], got shape {first_attn.shape}")
    
    actual_patch_tokens = first_attn.shape[2]
    
    # PRESERVE original per-tile values BEFORE any updates (critical for tiled mode)
    per_tile_patches_original = num_patches  # Should be 1600 per tile
    per_tile_grid_h_original = grid_h  # Should be 40
    per_tile_grid_w_original = grid_w  # Should be 40
    
    # Check if we have tiled mode and handle accordingly
    # Get vision_meta if not already available
    if 'vision_meta' not in locals() or vision_meta is None:
        vision_meta = None
        try:
            with torch.no_grad():
                if len(pixel_values.shape) == 6:
                    pv_test = pixel_values[:1, :, :, :, :, :]
                else:
                    pv_test = pixel_values[:1]
                test_vision_out = vision_model(pv_test, return_dict=True, 
                                             aspect_ratio_ids=aspect_ratio_ids[:1] if (aspect_ratio_ids is not None and aspect_ratio_ids.dim() > 0) else aspect_ratio_ids,
                                             aspect_ratio_mask=aspect_ratio_mask[:1] if (aspect_ratio_mask is not None and aspect_ratio_mask.dim() > 0) else aspect_ratio_mask)
                if hasattr(test_vision_out, 'last_hidden_state'):
                    _, vision_meta = normalize_vision_hidden_state(test_vision_out.last_hidden_state)
        except Exception:
            pass
    
    if vision_meta and vision_meta['Ntiles'] > 1:
        # Tiled mode: actual_patch_tokens is TOTAL across all tiles
        per_tile_expected = num_patches  # This is per-tile from detection (1600)
        total_expected = per_tile_expected * vision_meta['Ntiles']  # 4 × 1600 = 6400
        
        if actual_patch_tokens >= total_expected:
            # Keep per-tile values, will extract per-tile in heatmap computation
            # Store original per-tile values (these are the correct per-tile dimensions)
            # DO NOT update num_patches/grid_h/grid_w - keep them as per-tile values
            per_tile_patches_original = num_patches  # 1600
            per_tile_grid_h_original = grid_h  # 40
            per_tile_grid_w_original = grid_w  # 40
        else:
            # Try to recompute per-tile from total
            per_tile_actual = actual_patch_tokens // vision_meta['Ntiles']
            if per_tile_actual * vision_meta['Ntiles'] == actual_patch_tokens:
                # Perfect division
                num_patches = per_tile_actual
                # Find grid for per-tile
                found_grid = False
                for h in range(int(math.sqrt(per_tile_actual)), 0, -1):
                    w = per_tile_actual // h
                    if h * w == per_tile_actual:
                        grid_h, grid_w = h, w
                        found_grid = True
                        break
                if not found_grid:
                    raise ValueError(f"Cannot determine per-tile grid from {per_tile_actual} patches")
            else:
                raise ValueError(
                    f"Cannot divide {actual_patch_tokens} tokens evenly into {vision_meta['Ntiles']} tiles. "
                    f"Expected {total_expected} total ({per_tile_expected} per tile)."
                )
    else:
        # Non-tiled mode: should match exactly
        if actual_patch_tokens != num_patches:
            # Try to find valid grid
            found_grid = False
            for h in range(int(math.sqrt(actual_patch_tokens)), 0, -1):
                w = actual_patch_tokens // h
                if h * w == actual_patch_tokens:
                    grid_h, grid_w = h, w
                    num_patches = actual_patch_tokens
                    found_grid = True
                    break
            
            if not found_grid:
                raise ValueError(
                    f"Cannot determine patch grid from patch count {actual_patch_tokens}. "
                    f"Expected: {num_patches}."
                )
    
    # Verify all captured tensors have the same patch token count
    for i, attn in enumerate(captured_attentions):
        if len(attn.shape) != 3:
            raise ValueError(f"Expected [B, H, patch_tokens], got shape {attn.shape}")
        if attn.shape[2] != actual_patch_tokens:
            pass
    
    # Get vision_meta if not already available (from capture verification)
    if 'vision_meta' not in locals() or vision_meta is None:
        vision_meta = None
        try:
            with torch.no_grad():
                # Handle 6D pixel_values: [B, Nimg, Ntiles, C, H, W]
                if len(pixel_values.shape) == 6:
                    pv_test = pixel_values[:1, :, :, :, :, :]
                else:
                    pv_test = pixel_values[:1]
                test_vision_out = vision_model(pv_test, return_dict=True, 
                                             aspect_ratio_ids=aspect_ratio_ids[:1] if aspect_ratio_ids is not None and aspect_ratio_ids.dim() > 0 else aspect_ratio_ids,
                                             aspect_ratio_mask=aspect_ratio_mask[:1] if aspect_ratio_mask is not None and aspect_ratio_mask.dim() > 0 else aspect_ratio_mask)
                if hasattr(test_vision_out, 'last_hidden_state'):
                    _, vision_meta = normalize_vision_hidden_state(test_vision_out.last_hidden_state)
        except:
            pass
    
    # Compute attention heatmap (using robust grid - NO PADDING/TRIMMING)
    
    # Check if we need to handle tiles
    Ntiles = vision_meta['Ntiles'] if vision_meta and vision_meta['Ntiles'] > 1 else 1
    heatmaps_per_tile = None
    
    # ========================================================================
    # A) VALID TILE DETECTION (CRITICAL)
    # ========================================================================
    valid_tiles = list(range(Ntiles))  # Default: all tiles valid
    if Ntiles > 1:
        if aspect_ratio_mask is not None:
            # aspect_ratio_mask shape is typically [B, Nimg, Ntiles] or [B, Ntiles]
            mask_flat = aspect_ratio_mask.flatten()
            if len(mask_flat) >= Ntiles:
                # Extract per-tile validity
                valid_tiles = []
                for tile_idx in range(Ntiles):
                    if tile_idx < len(mask_flat):
                        is_valid = mask_flat[tile_idx].item() > 0 if mask_flat.dtype != torch.bool else mask_flat[tile_idx].item()
                        if is_valid:
                            valid_tiles.append(tile_idx)
                
                if not valid_tiles:
                    valid_tiles = list(range(Ntiles))
    
    if Ntiles > 1:
        
        # The captured attention is [1, H, total_patch_tokens] where total_patch_tokens includes all tiles
        # From debug: seq_len=6432, so total tokens = 6432
        # We know per_tile_T = 1601, so 4 tiles × 1601 = 6404 tokens
        # The extra 28 tokens might be register/global tokens
        
        per_tile_T = vision_meta['T']  # 1601 per tile
        # Use ORIGINAL per-tile values (before any updates that might have changed num_patches to total)
        per_tile_patches = per_tile_patches_original if 'per_tile_patches_original' in locals() else num_patches        
        # Calculate token ranges for each tile in the flattened sequence
        # Structure: [CLS_0, patches_0, CLS_1, patches_1, CLS_2, patches_2, CLS_3, patches_3, ...]
        # Or possibly: [global_CLS, tile_0_tokens, tile_1_tokens, ...]
        # From the debug output, we see seq_len=6432 and n_special=1, so likely:
        # Token 0: CLS (global or tile 0)
        # Tokens 1-6431: All patch tokens (all tiles combined)
        
        # Try to extract per-tile by assuming tiles are arranged sequentially
        # Each tile has per_tile_T tokens, so:
        # Tile 0: tokens [0:per_tile_T] = [0:1601] -> patches [1:1601] (skip CLS at 0)
        # Tile 1: tokens [per_tile_T:2*per_tile_T] = [1601:3202] -> patches [1602:3202] (skip CLS at 1601)
        # etc.
        
        heatmaps_per_tile = []
        for tile_idx in range(Ntiles):
            # Calculate token range for this tile
            tile_token_start = tile_idx * per_tile_T
            tile_token_end = (tile_idx + 1) * per_tile_T
            # Patches start after CLS (which is at tile_token_start)
            patch_start_in_flattened = tile_token_start + 1  # Skip CLS
            patch_end_in_flattened = tile_token_end
            
            # But the captured attention is [1, H, total_patch_tokens] where total_patch_tokens
            # starts AFTER n_special (which is 1, the first CLS)
            # So in the captured attention, tile 0 patches start at index 0, tile 1 at per_tile_patches, etc.
            patch_start_in_captured = tile_idx * per_tile_patches
            patch_end_in_captured = (tile_idx + 1) * per_tile_patches
            
            # Extract attention for this tile from captured attention
            tile_attentions = []
            for attn in captured_attentions:
                # attn is [1, H, total_patch_tokens] - extract this tile's patch tokens
                if attn.shape[2] >= patch_end_in_captured:
                    tile_patch_attn = attn[:, :, patch_start_in_captured:patch_end_in_captured]  # [1, H, per_tile_patches]
                    tile_attentions.append(tile_patch_attn)
                else:
                    # Use zeros as fallback
                    tile_patch_attn = torch.zeros(1, attn.shape[1], per_tile_patches, dtype=attn.dtype)
                    tile_attentions.append(tile_patch_attn)
            
            # Compute heatmap for this tile using PER-TILE grid (40×40, not 59×109)
            # The grid_h/grid_w should still be per-tile (40×40) if we preserved them correctly
            # But if they were updated, use the original per-tile values
            if 'per_tile_grid_h_original' in locals():
                per_tile_grid_h = per_tile_grid_h_original
                per_tile_grid_w = per_tile_grid_w_original
            else:
                # Fallback: assume current grid_h/grid_w are per-tile (should be 40×40)
                per_tile_grid_h = grid_h
                per_tile_grid_w = grid_w
            
            if args.mode == "last":
                tile_heatmap = compute_last_block_attention(tile_attentions, per_tile_grid_h, per_tile_grid_w)
            else:
                tile_heatmap = compute_attention_rollout(tile_attentions, per_tile_grid_h, per_tile_grid_w)
            
            heatmaps_per_tile.append(tile_heatmap)
        
        # ========================================================================
        # B) REPRESENTATIVE TILE SELECTION
        # ========================================================================
        # Compute attention metrics for valid tiles only
        tile_metrics = []
        for tile_idx in valid_tiles:
            if tile_idx < len(heatmaps_per_tile):
                tile_heatmap = heatmaps_per_tile[tile_idx]
                heatmap_flat = tile_heatmap.flatten()
                valid_mask = np.isfinite(heatmap_flat) & (heatmap_flat > 0)
                
                if valid_mask.sum() == 0:
                    # Skip tiles with no valid attention
                    continue
                
                valid_values = heatmap_flat[valid_mask]
                # Normalize to probability distribution
                valid_sum = valid_values.sum()
                if valid_sum > 0:
                    probs = valid_values / valid_sum
                    
                    # Compute entropy (lower = more focused)
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    
                    # Compute top-5% mass (higher = more salient)
                    top_5_percent_idx = int(len(valid_values) * 0.95)
                    top_5_percent_mass = np.sort(valid_values)[top_5_percent_idx:].sum() / valid_sum if valid_sum > 0 else 0.0
                    
                    tile_metrics.append({
                        'tile_idx': tile_idx,
                        'entropy': entropy,
                        'top_5_percent_mass': top_5_percent_mass,
                        'min': float(valid_values.min()),
                        'max': float(valid_values.max()),
                        'mean': float(valid_values.mean())
                    })
        
        # Select representative tile: maximize top-5% mass, break ties by minimizing entropy
        if tile_metrics:
            tile_metrics.sort(key=lambda x: (-x['top_5_percent_mass'], x['entropy']))
            representative_tile = tile_metrics[0]['tile_idx']
            
            # Combine valid tiles for main overlay (sum all valid tiles)
            if len(valid_tiles) > 1:
                combined_heatmap = np.zeros_like(heatmaps_per_tile[valid_tiles[0]])
                for tile_idx in valid_tiles:
                    if tile_idx < len(heatmaps_per_tile):
                        combined_heatmap += heatmaps_per_tile[tile_idx]
                heatmap = combined_heatmap
            else:
                # Use representative tile for main overlay (single valid tile)
                heatmap = heatmaps_per_tile[representative_tile]
        else:
            # Fallback: use first valid tile
            representative_tile = valid_tiles[0] if valid_tiles else 0
            heatmap = heatmaps_per_tile[representative_tile]
    else:
        # Non-tiled mode: standard processing
        if args.mode == "last":
            heatmap = compute_last_block_attention(captured_attentions, grid_h, grid_w)
        else:
            heatmap = compute_attention_rollout(captured_attentions, grid_h, grid_w)
    
    # ========================================================================
    # C) VISUALIZATION PARAMETER TUNING (Emphasize salient regions)
    # ========================================================================
    # Override defaults to reduce diffusion and emphasize salient regions
    # These parameters apply to both tiled and non-tiled modes
    viz_clip_low = max(args.clip_low, 50.0)  # Increase lower percentile (default 50 if not set)
    viz_clip_high = min(args.clip_high, 99.5)  # Keep high percentile reasonable
    viz_gamma = max(0.9, min(args.gamma, 1.0))  # Gamma close to 1 (0.9-1.0)
    viz_sigma = min(args.sigma, 1.0)  # Reduce smoothing (default 1.0 max)
    
    # Initialize representative_tile for non-tiled mode
    if Ntiles == 1:
        representative_tile = 0
    
    # Create improved overlay
    visualizations = create_improved_overlay(
        sample['image'],
        heatmap,
        grid_h,
        grid_w,
        alpha=args.alpha,
        colormap=args.colormap,
        clip_percentiles=(viz_clip_low, viz_clip_high),
        gamma=viz_gamma,
        sigma=viz_sigma,
        border_mask_ratio=args.border_mask
    )
    
    # Save outputs
    output_base = Path(args.out_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Compute gradient attribution if requested
    # ========================================================================
    gradient_heatmap = None
    if args.do_gradient:
        print("Computing gradient attribution...")
        try:
            # Clean up previous batch to free memory
            del batch
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Recreate sample using VLM_Dataset (same as gradient_attribution_vis.py)
            # This ensures sample has 'chat_template' key required by custom_data_collator
            data = load_data(args.test_data_path, args.summary_type)
            sample_item = next((item for item in data if item['id'] == args.unique_id), None)
            if sample_item is None:
                raise ValueError(f"Sample not found: {args.unique_id}")
            
            # Create ModelArgs instance for VLM_Dataset
            model_args_for_dataset = ModelArgs()
            model_args_for_dataset.test_metadata_image_path = args.metadata_image_path
            
            dataset = VLM_Dataset(
                model_args_for_dataset, [sample_item], args.metadata_image_path,
                use_cxr_image=True, use_rad_report=True, use_generated_rad_report=True, use_discharge_note=True, shuffle=False
            )
            sample_for_grad = dataset[0]
            
            # Recreate batch for gradient computation (to avoid gradient history)
            from dataloader import custom_data_collator
            batch_grad = custom_data_collator(processor, use_cxr_image=True, summary_type=args.summary_type)([sample_for_grad])
            batch_grad = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_grad.items()}
            
            gradient_heatmap = compute_gradient_attribution(
                model, batch_grad, device,
                target_class=1,
                grid_h=grid_h,
                grid_w=grid_w,
                cls_index=cls_index,
                n_special=n_special,
                num_patches=num_patches,
                representative_tile=representative_tile if Ntiles > 1 else 0,
                vision_meta=vision_meta
            )
            
            # Clean up batch immediately
            del batch_grad
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception:
            gradient_heatmap = None
    
    # ========================================================================
    # Compute occlusion sensitivity if requested
    # ========================================================================
    occlusion_heatmap = None
    base_death_logit_occ = None
    if args.do_occlusion:
        print("Computing occlusion sensitivity...")
        try:
            # Recreate sample using VLM_Dataset (same as gradient_attribution_vis.py)
            # This ensures sample has 'chat_template' key required by custom_data_collator
            data = load_data(args.test_data_path, args.summary_type)
            sample_item = next((item for item in data if item['id'] == args.unique_id), None)
            if sample_item is None:
                raise ValueError(f"Sample not found: {args.unique_id}")
            
            # Create ModelArgs instance for VLM_Dataset
            model_args_for_dataset = ModelArgs()
            model_args_for_dataset.test_metadata_image_path = args.metadata_image_path
            
            dataset = VLM_Dataset(
                model_args_for_dataset, [sample_item], args.metadata_image_path,
                use_cxr_image=True, use_rad_report=True, use_generated_rad_report=True, use_discharge_note=True, shuffle=False
            )
            sample_for_occ = dataset[0]
            
            # Recreate batch for occlusion computation (to avoid any gradient history)
            from dataloader import custom_data_collator
            batch_occ = custom_data_collator(processor, use_cxr_image=True, summary_type=args.summary_type)([sample_for_occ])
            batch_occ = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_occ.items()}
            
            occlusion_heatmap, base_death_logit_occ = compute_occlusion_sensitivity(
                model, batch_occ, device, processor, sample_for_occ,
                target_class=1,
                grid_h=grid_h,
                grid_w=grid_w,
                representative_tile=representative_tile if Ntiles > 1 else 0,
                vision_meta=vision_meta,
                batch_size=args.occlusion_batch_size,
                mask_value=args.occlusion_mask_value,
                occlusion_mode=args.occlusion_mode
            )
            
            # Clean up batch immediately
            del batch_occ
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception:
            occlusion_heatmap = None
            base_death_logit_occ = None
    
    # ========================================================================
    # Create 4-panel comparison figure
    # ========================================================================
    output_path = output_base / f"{args.unique_id}_comparison.png"
    create_single_model_comparison_figure(
        sample['image'],
        heatmap,
        gradient_heatmap,
        occlusion_heatmap,
        prediction,
        prob_death,
        str(output_path),
        args.unique_id,
        args
    )
    
    # Save summary JSON
    summary = {
        "unique_id": args.unique_id,
        "discharge_note": sample.get('discharge_note'),
        "radiology_report": sample.get('radiology_report'),
        "generated_radiology_report": sample.get('generated_radiology_report'),
        "image_path": sample.get('image_path'),
        "prob_class_1": prob_death,
    }
    
    summary_path = output_base / f"{args.unique_id}_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    


if __name__ == "__main__":
    main()
