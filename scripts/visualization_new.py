#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cxr_vlm_explain_3stage.py
3-stage explainability for Mllama (Llama 3.2 Vision Instruct 11B) mortality prediction:
1) Vision self-attn: CLS -> patch (query-only, 1xT)
2) LLM cross-attn: text token -> image tokens Grad*Attn (Chefer-style, query-only, 1xS_img)
3) Signed occlusion: Δlogit = base - masked (red supports positive, blue supports negative)
"""
import os
import sys
import gc
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm
from transformers import AutoProcessor

# ---- import your codebase modules ----
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from dataloader import load_hash2meta_dict, CXRDecisionTree
from model import load_model

# -------------------------
# Utils: memory
# -------------------------
def hard_cuda_gc(device: torch.device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

# -------------------------
# Resolve submodules
# -------------------------
def resolve_base_and_vision(model: nn.Module):
    base = getattr(model, "base_model", model)
    if hasattr(base, "vision_model"):
        return base, base.vision_model
    if hasattr(model, "vision_model"):
        return model, model.vision_model
    for cand in ["vision", "vision_encoder", "visual", "image_encoder"]:
        if hasattr(base, cand):
            return base, getattr(base, cand)
        if hasattr(model, cand):
            return model, getattr(model, cand)
    raise AttributeError("Cannot resolve vision model.")

def resolve_language_model(model: nn.Module):
    base = getattr(model, "base_model", model)
    if hasattr(base, "language_model"):
        return base.language_model
    if hasattr(model, "language_model"):
        return model.language_model
    for cand in ["model", "lm", "text_model"]:
        if hasattr(base, cand):
            return getattr(base, cand)
        if hasattr(model, cand):
            return getattr(model, cand)
    raise AttributeError("Cannot resolve language model.")

def resolve_projector(model: nn.Module):
    base = getattr(model, "base_model", model)
    if hasattr(base, "multi_modal_projector"):
        return base.multi_modal_projector
    if hasattr(model, "multi_modal_projector"):
        return model.multi_modal_projector
    for cand in ["mm_projector", "projector"]:
        if hasattr(base, cand):
            return getattr(base, cand)
        if hasattr(model, cand):
            return getattr(model, cand)
    return None

# -------------------------
# Normalize hidden state shapes (3D/4D/5D)
# -------------------------
def normalize_vision_hidden_state(hs: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
    orig_shape = list(hs.shape)
    ndim = len(orig_shape)
    if ndim == 3:
        B, T, D = orig_shape
        meta = {"layout": "BTD", "B": int(B), "Nimg": 1, "Ntiles": 1, "T": int(T), "D": int(D)}
        return hs, meta
    if ndim == 4:
        B, Ntiles, T, D = orig_shape
        hs_flat = hs.view(B * Ntiles, T, D)
        meta = {"layout": "BNTD", "B": int(B), "Nimg": 1, "Ntiles": int(Ntiles), "T": int(T), "D": int(D)}
        return hs_flat, meta
    if ndim == 5:
        B, Nimg, Ntiles, T, D = orig_shape
        hs_flat = hs.view(B * Nimg * Ntiles, T, D)
        meta = {"layout": "BINTD", "B": int(B), "Nimg": int(Nimg), "Ntiles": int(Ntiles), "T": int(T), "D": int(D)}
        return hs_flat, meta
    raise ValueError(f"Unsupported vision hidden state shape: {orig_shape}")

# -------------------------
# Robust: detect patch grid + token layout (per tile)
# -------------------------
def detect_vision_tokens_and_grid(
    vision_model: nn.Module,
    pixel_values: torch.Tensor,
    processor: AutoProcessor,
    aspect_ratio_ids: Optional[torch.Tensor] = None,
    aspect_ratio_mask: Optional[torch.Tensor] = None,
) -> Tuple[int, int, int, int, int, Optional[Dict]]:
    config = vision_model.config
    image_size = getattr(config, "image_size", None)
    patch_size = getattr(config, "patch_size", None)
    if isinstance(image_size, (list, tuple)):
        image_size = image_size[0] if image_size else None
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[0] if patch_size else None

    if image_size is None or patch_size is None:
        if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size"):
            s = processor.image_processor.size
            if isinstance(s, dict):
                image_size = image_size or s.get("height") or s.get("shortest_edge")
            elif isinstance(s, (int, list, tuple)):
                image_size = image_size or (s if isinstance(s, int) else s[0])
        if patch_size is None:
            patch_size = 14 if (image_size and image_size >= 336) else 16

    if not image_size or not patch_size:
        raise ValueError(f"Cannot infer image_size/patch_size. image_size={image_size}, patch_size={patch_size}")

    grid_h = grid_w = image_size // patch_size
    expected_patches = grid_h * grid_w
    vision_meta = None
    seq_len = None

    try:
        with torch.no_grad():
            kwargs = {"pixel_values": pixel_values[:1]}
            if aspect_ratio_ids is not None:
                kwargs["aspect_ratio_ids"] = aspect_ratio_ids[:1] if aspect_ratio_ids.dim() > 0 else aspect_ratio_ids
            if aspect_ratio_mask is not None:
                kwargs["aspect_ratio_mask"] = aspect_ratio_mask[:1] if aspect_ratio_mask.dim() > 0 else aspect_ratio_mask
            out = vision_model(**kwargs)
            hs = getattr(out, "last_hidden_state", None)
            if hs is None and isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
                hs = out[0]
            if hs is None:
                raise RuntimeError("Cannot extract vision hidden state for token/grid detection.")
            _, vision_meta = normalize_vision_hidden_state(hs)
            seq_len = int(vision_meta["T"])
    except Exception:
        seq_len = expected_patches + 1
        vision_meta = None

    cls_index = 0
    n_special = seq_len - expected_patches
    if n_special < 1:
        raise ValueError(f"Invalid n_special={n_special}. seq_len={seq_len}, expected_patches={expected_patches}")

    return cls_index, n_special, expected_patches, grid_h, grid_w, vision_meta

def find_mllama_cross_attn_modules(model: nn.Module):
    """
    Return list of (name, module) for all Mllama cross-attention modules.
    We match by module name containing '.cross_attn' (works with modeling_mllama stacktrace)
    """
    mods = []
    for name, mod in model.named_modules():
        if name.endswith(".cross_attn"):
            mods.append((name, mod))
    return mods


def _extract_cross_states_from_args_kwargs(args, kwargs):
    """
    Mllama cross_attn forward signature in HF typically is:
      forward(hidden_states, attention_mask=None, position_ids=None,
              past_key_value=None, output_attentions=False, use_cache=False,
              cache_position=None, cross_attention_states=None, cross_attention_mask=None, ...)
    But to be robust, we try:
      - kwargs common keys
      - then positional tensors after hidden_states
    """
    # 1) kwargs keys
    for key in [
        "cross_attention_states",
        "encoder_hidden_states",
        "key_value_states",
        "vision_hidden_states",
        "image_hidden_states",
    ]:
        v = kwargs.get(key, None)
        if torch.is_tensor(v) and v.dim() == 3:
            return v

    # 2) positional scan (skip args[0]=hidden_states)
    for a in args[1:]:
        if torch.is_tensor(a) and a.dim() == 3:
            return a

    return None


def install_mllama_cross_attn_score_capture(
    model: nn.Module,
    token_index: int,
    capture_last_n: int,
    storage: List[torch.Tensor],
):
    """
    Capture pre-softmax attention scores row for ONE text token:
      scores = (q_tok @ k_img^T) / sqrt(d)  -> [B,H,Simg]
    Keep it on GPU and retain_grad() so we can use grad(scores) after backward.
    This works even if the model uses SDPA internally (we compute scores from the same q/k projections).
    """
    all_cross = find_mllama_cross_attn_modules(model)
    if not all_cross:
        raise RuntimeError("No modules named '*.cross_attn' found in model.named_modules().")

    selected = all_cross[-capture_last_n:] if capture_last_n > 0 else all_cross
    originals: List[Tuple[nn.Module, callable]] = []

    def make_wrapper(cross_mod: nn.Module, orig_forward, tag_name: str):
        q_proj = getattr(cross_mod, "q_proj", None)
        k_proj = getattr(cross_mod, "k_proj", None)
        num_heads = getattr(cross_mod, "num_heads", getattr(cross_mod, "num_attention_heads", None))

        def wrapped_forward(hidden_states, *args, **kwargs):
            out = orig_forward(hidden_states, *args, **kwargs)

            if q_proj is None or k_proj is None:
                return out
            if (not torch.is_tensor(hidden_states)) or hidden_states.dim() != 3:
                return out

            cross_states = _extract_cross_states_from_args_kwargs(args, kwargs)
            if cross_states is None:
                return out

            try:
                B, Ttxt, E = hidden_states.shape
                Simg = cross_states.shape[1]
                H = int(num_heads) if num_heads is not None else max(1, E // 64)
                d = E // H
                if d * H != E:
                    return out

                tidx = int(token_index)
                if tidx < 0:
                    tidx = Ttxt - 1
                tidx = max(0, min(tidx, Ttxt - 1))

                # IMPORTANT: do NOT detach/float in a way that breaks graph.
                # Keep computation on the graph; only cast to fp32 (still differentiable).
                q = q_proj(hidden_states[:, tidx:tidx + 1, :])  # [B,1,E]
                k = k_proj(cross_states)                       # [B,Simg,E]

                q = q.view(B, 1, H, d).transpose(1, 2)         # [B,H,1,d]
                k = k.view(B, Simg, H, d).transpose(1, 2)      # [B,H,Simg,d]

                # pre-softmax scores that are fully connected to model graph via q_proj/k_proj
                scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (1.0 / math.sqrt(d))  # [B,H,1,Simg]
                scores = scores[:, :, 0, :]  # [B,H,Simg]
                scores.retain_grad()
                storage.append(scores)
            except Exception:
                pass

            return out

        return wrapped_forward

    for (name, cross_mod) in selected:
        originals.append((cross_mod, cross_mod.forward))
        cross_mod.forward = make_wrapper(cross_mod, cross_mod.forward, name)

    print(f"[OK] Patched cross-attn modules: {len(selected)}/{len(all_cross)} (last_n={capture_last_n})")
    return originals


# -------------------------
# Patch restore helper (supports list/dict)
# -------------------------
def restore_patched_forwards(originals):
    if originals is None:
        return
    if isinstance(originals, dict):
        for module, orig_fwd in originals.values():
            module.forward = orig_fwd
        return
    for module, orig_fwd in originals:
        module.forward = orig_fwd

# -------------------------
# Stage 1) Vision self-attn: CLS -> patch (query-only, 1xT)
# -------------------------
def install_vit_cls_row_capture(
    vision_model: nn.Module,
    cls_index: int,
    n_special: int,
    capture_last_n: int,
    storage: list,
    debug_once: bool = True,
):
    if not hasattr(vision_model, "transformer") or not hasattr(vision_model.transformer, "layers"):
        raise ValueError("vision_model.transformer.layers not found")

    layers = vision_model.transformer.layers
    L = len(layers)
    capture_indices = list(range(max(0, L - capture_last_n), L))
    originals = []
    printed = {"done": False}

    def wrapped_factory(attn_module, layer_idx, original_forward):
        def wrapped_forward(*args, **kwargs):
            out = original_forward(*args, **kwargs)
            if layer_idx not in capture_indices:
                return out
            try:
                hidden_states = None
                attention_mask = None
                if len(args) >= 1 and torch.is_tensor(args[0]):
                    hidden_states = args[0]
                else:
                    hidden_states = kwargs.get("hidden_states", None) or kwargs.get("x", None)
                if hidden_states is None or not torch.is_tensor(hidden_states):
                    return out

                if "attention_mask" in kwargs:
                    attention_mask = kwargs["attention_mask"]
                elif "attn_mask" in kwargs:
                    attention_mask = kwargs["attn_mask"]
                elif len(args) >= 2 and torch.is_tensor(args[1]):
                    attention_mask = args[1]

                B, T, D = hidden_states.shape
                num_heads = getattr(attn_module, "num_heads", None) or getattr(attn_module, "num_attention_heads", None)
                if num_heads is None and hasattr(attn_module, "config"):
                    num_heads = getattr(attn_module.config, "num_attention_heads", None)
                if num_heads is None:
                    num_heads = 20 if D % 20 == 0 else max(1, D // 64)
                head_dim = D // num_heads

                q_proj = getattr(attn_module, "q_proj", None)
                k_proj = getattr(attn_module, "k_proj", None)
                if q_proj is None or k_proj is None:
                    return out

                q = q_proj(hidden_states)
                k = k_proj(hidden_states)
                q = q.view(B, T, num_heads, head_dim).transpose(1, 2)
                k = k.view(B, T, num_heads, head_dim).transpose(1, 2)

                q_cls = q[:, :, cls_index:cls_index + 1, :]
                scores = torch.matmul(q_cls.float(), k.float().transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))

                if attention_mask is not None:
                    am = attention_mask
                    if am.dim() == 2:
                        am = am[:, None, None, :]
                    elif am.dim() == 3:
                        am = am[:, None, :, :]
                        am = am[:, :, cls_index:cls_index+1, :]
                    elif am.dim() == 4:
                        am = am[:, :1, cls_index:cls_index+1, :]
                    if am.dtype == torch.bool:
                        scores = scores.masked_fill(~am, float("-inf"))
                    else:
                        scores = scores + am

                probs = torch.softmax(scores, dim=-1)
                cls_row = probs[:, :, 0, :]
                cls_to_patches = cls_row[:, :, n_special:]
                storage.append(cls_to_patches.detach().float().cpu())
            except Exception as e:
                if debug_once and not printed["done"]:
                    printed["done"] = True
                    print(f"[WARN] CLS-row capture failed at layer {layer_idx}: {type(e).__name__}: {e}")
            return out
        return wrapped_forward

    for i in capture_indices:
        layer = layers[i]
        if not hasattr(layer, "self_attn"):
            continue
        attn = layer.self_attn
        originals.append((attn, attn.forward))
        attn.forward = wrapped_factory(attn, i, attn.forward)

    return originals

def cls_row_to_heatmap(
    cls_rows: List[torch.Tensor],
    grid_h: int,
    grid_w: int,
    mode: str = "last",
) -> np.ndarray:
    if not cls_rows:
        raise RuntimeError("No vision CLS rows captured.")
    if mode == "last":
        v = cls_rows[-1].mean(dim=1).mean(dim=0)
    else:
        rollout = None
        for a in cls_rows:
            x = a.mean(dim=1).mean(dim=0)
            x = x / (x.sum() + 1e-8)
            rollout = x if rollout is None else (rollout * x)
        rollout = rollout / (rollout.sum() + 1e-8)
        v = rollout

    P = grid_h * grid_w
    if v.numel() < P:
        raise ValueError(f"Captured patch tokens {v.numel()} < expected {P}")
    v = v[:P]
    return v.numpy().reshape(grid_h, grid_w)

# -------------------------
# Stage 2) LLM cross-attn: Grad * Attn (query-only)
# -------------------------
def find_text_layers(model: nn.Module) -> nn.ModuleList:
    candidates = [
        ("base_model", "language_model", "model", "layers"),
        ("base_model", "language_model", "layers"),
        ("language_model", "model", "layers"),
        ("model", "layers"),
    ]
    for path in candidates:
        m = model
        ok = True
        for p in path:
            if not hasattr(m, p):
                ok = False
                break
            m = getattr(m, p)
        if ok and isinstance(m, (nn.ModuleList, list)) and len(m) > 0:
            return m

    for name, mod in model.named_modules():
        if name.endswith("layers") and isinstance(mod, nn.ModuleList) and len(mod) > 0:
            return mod

    raise ValueError("Cannot locate text transformer layers (ModuleList).")

def get_cross_attn_module(layer: nn.Module) -> Optional[nn.Module]:
    for attr in ["cross_attn", "cross_attention", "image_cross_attn", "cross_attention_layer"]:
        if hasattr(layer, attr):
            return getattr(layer, attr)
    for n, m in layer.named_children():
        ln = n.lower()
        if "cross" in ln and "attn" in ln:
            return m
    return None

def install_cross_attn_row_capture_with_grad(
    model: nn.Module,
    token_index: int,
    capture_last_n: int,
    storage: List[torch.Tensor],
):
    """
    Capture attention row for ONE text token_index: probs_row = softmax(q_tok @ k_img^T) -> [B,H,S_img]
    Keep it on GPU + retain_grad().
    """
    layers = find_text_layers(model)
    L = len(layers)
    capture_idx = list(range(max(0, L - capture_last_n), L))
    originals: List[Tuple[nn.Module, callable]] = []

    def wrap_cross_forward(orig_forward, layer_idx, cross_attn_mod):
        q_proj = getattr(cross_attn_mod, "q_proj", None)
        k_proj = getattr(cross_attn_mod, "k_proj", None)
        num_heads = getattr(cross_attn_mod, "num_heads", getattr(cross_attn_mod, "num_attention_heads", None))

        def fwd(hidden_states, *args, **kwargs):
            out = orig_forward(hidden_states, *args, **kwargs)
            if layer_idx not in capture_idx:
                return out
            if q_proj is None or k_proj is None:
                return out
            if not torch.is_tensor(hidden_states) or hidden_states.dim() != 3:
                return out

            # --- resolve cross states robustly (kw or positional) ---
            for key in ["cross_attention_states", "encoder_hidden_states", "key_value_states", 
            "vision_hidden_states", "image_hidden_states"]:
                cross_states = kwargs.get(key, None)
                if cross_states is not None:
                    break
            if cross_states is None:
                for a in args:
                    if torch.is_tensor(a) and a.dim() == 3:
                        cross_states = a
                        break

            if cross_states is None or (not torch.is_tensor(cross_states)) or cross_states.dim() != 3:
                return out

            try:
                B, Ttxt, E = hidden_states.shape
                Simg = cross_states.shape[1]
                H = num_heads or max(1, E // 64)
                d = E // H
                if d * H != E:
                    return out

                tidx = int(token_index)
                if tidx < 0:
                    tidx = Ttxt - 1
                tidx = max(0, min(tidx, Ttxt - 1))

                # Compute in fp32 and keep on GPU
                q = q_proj(hidden_states[:, tidx:tidx + 1, :]).float()
                k = k_proj(cross_states).float()
                q = q.view(B, 1, H, d).transpose(1, 2)
                k = k.view(B, Simg, H, d).transpose(1, 2)

                scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
                probs = torch.softmax(scores, dim=-1)[:, :, 0, :]  # [B,H,Simg] fp32 on GPU
                
                # Important: keep on GPU and retain grad
                probs.retain_grad()
                storage.append(probs)
            except Exception:
                pass
            return out
        return fwd

    patched_count = 0
    for i in capture_idx:
        layer = layers[i]
        cross_mod = get_cross_attn_module(layer)
        if cross_mod is None:
            continue
        originals.append((cross_mod, cross_mod.forward))
        cross_mod.forward = wrap_cross_forward(cross_mod.forward, i, cross_mod)
        patched_count += 1

    if patched_count == 0:
        print("[WARN] No cross-attention modules were patched. Model may not have cross-attention in expected layers.")
        # Don't raise, just warn - model might still work

    return originals

def gradattn_rows_to_patch_heatmap(
    rows_with_grad: List[torch.Tensor],  # now these are scores [B,H,Simg] with grad
    grid_h: int,
    grid_w: int,
    n_special: int,
    vision_meta: Optional[Dict],
    per_tile_patches: int,
) -> np.ndarray:
    if not rows_with_grad:
        raise RuntimeError("No cross-attn rows captured for Grad*Attn.")

    agg = None
    for scores in rows_with_grad:
        if scores is None or (not torch.is_tensor(scores)):
            continue
        if scores.grad is None:
            continue

        # A = softmax(scores)  (attention weights)
        attn = torch.softmax(scores, dim=-1)          # [B,H,Simg]
        g = scores.grad                               # [B,H,Simg]

        # Chefer-style: relu(attn * grad) aggregated
        x = torch.relu(attn * g).mean(dim=1).mean(dim=0)  # [Simg]
        agg = x if agg is None else (agg + x)

    if agg is None:
        raise RuntimeError("No gradients found on captured cross-attn scores (did backward run?)")

    agg = agg.detach().float().cpu()
    S = agg.numel()
    P = grid_h * grid_w

    # map to patches (best-effort)
    if vision_meta and vision_meta.get("Ntiles", 1) > 1:
        Ntiles = int(vision_meta["Ntiles"])
        T_per_tile = int(vision_meta.get("T", n_special + per_tile_patches))
        patches_all = []
        for t in range(Ntiles):
            off = t * T_per_tile
            st = off + n_special
            ed = st + per_tile_patches
            if ed <= S:
                patches_all.append(agg[st:ed])
        if not patches_all:
            total_needed = Ntiles * P
            if S >= total_needed:
                patches_all = [agg[t * P:(t + 1) * P] for t in range(Ntiles)]
            else:
                raise ValueError(f"Cannot map cross-attn seq (S={S}) to tiles/patches.")
        v = torch.stack(patches_all, dim=0).sum(dim=0)
    else:
        if S >= n_special + P:
            v = agg[n_special:n_special + P]
        else:
            if S < P:
                raise ValueError(f"Cross-attn seq too short S={S}, expected >= {P}")
            v = agg[:P]

    v = v / (v.max() + 1e-8)
    return v.numpy().reshape(grid_h, grid_w)


# -------------------------
# Stage 3) Signed occlusion (micro-batch)
# -------------------------
@torch.inference_mode()
def compute_signed_occlusion_map(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    target_class: int,
    grid_h: int,
    grid_w: int,
    representative_tile: int = 0,
    stride: int = 2,
    mask_value: float = 0.0,
    micro_batch: int = 8,
) -> Tuple[np.ndarray, float]:
    model.eval()
    base_out = model(**batch)
    base_logit = float(base_out["logits"][0, target_class].item())

    pv = batch["pixel_values"]
    if pv.dim() == 6:
        # [B, Nimg, Ntiles, C, H, W]
        B, Nimg, Ntiles, C, H, W = pv.shape
        tile_indices = (0, representative_tile)  # (Nimg_idx, Ntile_idx)
    elif pv.dim() == 5:
        # [B, Ntiles, C, H, W]
        B, Ntiles, C, H, W = pv.shape
        tile_indices = (representative_tile,)
    elif pv.dim() == 4:
        # [B, C, H, W]
        B, C, H, W = pv.shape
        tile_indices = None
    else:
        raise ValueError(f"Unsupported pixel_values shape: {pv.shape}")

    patch_h = H // grid_h
    patch_w = W // grid_w
    block_h = patch_h * stride
    block_w = patch_w * stride
    step_h = grid_h // stride
    step_w = grid_w // stride

    coords = []
    for r in range(step_h):
        for c in range(step_w):
            rs = r * block_h
            re = min((r + 1) * block_h, H)
            cs = c * block_w
            ce = min((c + 1) * block_w, W)
            coords.append((r, c, rs, re, cs, ce))

    small = np.zeros((step_h, step_w), dtype=np.float32)

    for i0 in tqdm(range(0, len(coords), micro_batch), desc=f"Occlusion signed (stride={stride})"):
        chunk = coords[i0:i0 + micro_batch]
        batch_masked = {}
        for k, v in batch.items():
            if not isinstance(v, torch.Tensor):
                batch_masked[k] = v
                continue
            if k == "pixel_values":
                vrep = v.repeat(len(chunk), *([1] * (v.dim() - 1)))
                for j, (_, _, rs, re, cs, ce) in enumerate(chunk):
                    if pv.dim() == 6:
                        vrep[j:j+1, tile_indices[0], tile_indices[1], :, rs:re, cs:ce] = mask_value
                    elif pv.dim() == 5:
                        vrep[j:j+1, tile_indices[0], :, rs:re, cs:ce] = mask_value
                    else:  # dim == 4
                        vrep[j:j+1, :, rs:re, cs:ce] = mask_value
                batch_masked[k] = vrep
            else:
                batch_masked[k] = v.repeat(len(chunk), *([1] * (v.dim() - 1)))

        out = model(**{k: (vv.to(device) if isinstance(vv, torch.Tensor) else vv) for k, vv in batch_masked.items()})
        masked_logits = out["logits"][:, target_class].float().cpu().numpy()

        for j, (r, c, *_rest) in enumerate(chunk):
            masked_logit = float(masked_logits[j])
            small[r, c] = base_logit - masked_logit

        del batch_masked, out
        hard_cuda_gc(device)

    zoom = (grid_h / step_h, grid_w / step_w)
    occ = ndimage.zoom(small, zoom, order=0)
    return occ, base_logit

# -------------------------
# Visualization helpers
# -------------------------
def normalize_pos_map(hm: np.ndarray, clip_low=50.0, clip_high=99.5, gamma=0.9) -> np.ndarray:
    x = hm.astype(np.float32)
    flat = x.flatten()
    m = np.isfinite(flat) & (flat > 0)
    if m.sum() == 0:
        return np.zeros_like(x)
    v = flat[m]
    lo = np.percentile(v, clip_low)
    hi = np.percentile(v, clip_high)
    if hi <= lo:
        hi = v.max()
        lo = v.min()
    y = np.clip(x, lo, hi)
    y = (y - lo) / (hi - lo + 1e-8)
    y = np.power(y, gamma)
    return y

def normalize_signed_map(hm: np.ndarray, clip_abs_percentile=99.0) -> Tuple[np.ndarray, float]:
    x = hm.astype(np.float32)
    s = np.percentile(np.abs(x[np.isfinite(x)]), clip_abs_percentile) if np.isfinite(x).any() else 1.0
    s = max(s, 1e-8)
    y = np.clip(x / s, -1.0, 1.0)
    return y, s

def overlay_heatmap_on_image(
    image: Image.Image,
    heat: np.ndarray,
    cmap: str,
    alpha: float,
    signed: bool = False,
) -> np.ndarray:
    img = np.array(image).astype(np.float32) / 255.0
    H, W = img.shape[:2]
    hm = heat
    if hm.shape != (H, W):
        hm = ndimage.zoom(hm, (H / hm.shape[0], W / hm.shape[1]), order=1 if not signed else 0)
    cm = plt.get_cmap(cmap)
    colored = cm((hm + 1) / 2.0)[:, :, :3] if signed else cm(hm)[:, :, :3]
    out = (1 - alpha) * img + alpha * colored
    out = np.clip(out * 255, 0, 255).astype(np.uint8)
    return out

def save_4panel(
    out_path: str,
    image: Image.Image,
    hm_vit: np.ndarray,
    hm_gradattn: np.ndarray,
    hm_occ_signed: np.ndarray,
    meta_title: str,
    alpha: float = 0.35,
    cmap_pos: str = "magma",
    cmap_signed: str = "seismic",
):
    img = np.array(image).astype(np.float32) / 255.0
    vit_norm = normalize_pos_map(hm_vit)
    ga_norm = normalize_pos_map(hm_gradattn)
    occ_norm, occ_scale = normalize_signed_map(hm_occ_signed)

    vit_overlay = overlay_heatmap_on_image(image, vit_norm, cmap_pos, alpha, signed=False)
    ga_overlay = overlay_heatmap_on_image(image, ga_norm, cmap_pos, alpha, signed=False)
    occ_overlay = overlay_heatmap_on_image(image, occ_norm, cmap_signed, alpha, signed=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img); axes[0].axis("off"); axes[0].set_title(f"Original\n{meta_title}", fontsize=9)
    axes[1].imshow(vit_overlay); axes[1].axis("off"); axes[1].set_title("ViT self-attn (CLS→patch)", fontsize=9)
    axes[2].imshow(ga_overlay); axes[2].axis("off"); axes[2].set_title("LLM cross-attn Grad*Attn", fontsize=9)
    axes[3].imshow(occ_overlay); axes[3].axis("off"); axes[3].set_title(f"Signed Occlusion (p99|Δ|={occ_scale:.3g})", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

# -------------------------
# Data loading (your original logic kept)
# -------------------------
def load_sample_by_unique_id(unique_id: str, args, processor: AutoProcessor) -> Dict:
    hash2meta = load_hash2meta_dict(args.metadata_path, args.metadata_image_path)
    if unique_id not in hash2meta:
        raise ValueError(f"Unique ID {unique_id} not found in metadata")

    discharge_note = None
    label = None
    try:
        with open(args.test_data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item.get("id") == unique_id:
                    if args.summary_type in item:
                        t = item[args.summary_type]
                        discharge_note = t[0] if isinstance(t, list) else t
                    elif "text" in item:
                        t = item["text"]
                        discharge_note = t[0] if isinstance(t, list) else t
                    if "label" in item:
                        label = item["label"]
                    break
    except Exception:
        pass

    decision_tree = CXRDecisionTree()
    all_img_paths = hash2meta[unique_id]["metadata_filtered"]
    selected = decision_tree.select_best_cxr(all_img_paths)
    if selected is None:
        raise ValueError(f"No CXR image found for unique_id {unique_id}")

    selected_img_data_path = selected[1]
    mp = args.metadata_image_path.lower()
    if "train" in mp:
        split = "train"
    elif "dev" in mp or "val" in mp:
        split = "dev"
    else:
        split = "test"

    image_path = selected_img_data_path.split("/")[-1]
    name, ext = image_path.split(".")
    if "_512_resized" in name:
        real_path = os.path.join(args.base_img_dir, split, image_path)
    else:
        real_path = os.path.join(args.base_img_dir, split, f"{name}_512_resized.{ext}")

    if not os.path.exists(real_path):
        raise FileNotFoundError(f"Image not found: {real_path}")

    img = Image.open(real_path).convert("RGB")
    
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
    
    return {
        "id": unique_id,
        "discharge_note": discharge_note,
        "radiology_report": rad_report,
        "generated_radiology_report": generated_rad_report,
        "label": label,
        "image": img,
        "image_path": real_path,
    }

# -------------------------
# Build batch
# -------------------------
def build_batch(processor: AutoProcessor, sample: Dict, device: torch.device) -> Tuple[Dict, List[Dict]]:
    system_prompt = (
        "A clinical document and a single, most recent chest X-ray (CXR) image from the patient are provided. "
        "Based on the clinical context and the provided CXR image, assess how likely the patient's out-of-hospital mortality is within 30 days."
    )

    if sample.get("discharge_note"):
        user_prompt = (
            f"Here is the clinical document:\n{sample['discharge_note']}\n\n"
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
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False)
    # processed = processor(text=[text], images=[[sample["image"]]], return_tensors="pt", padding=True)
    processed = processor(
        text=[text],
        images=[[sample["image"]]],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )

    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in processed.items()}
    return batch, messages

def find_all_cross_attn_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    mods = []
    for name, m in model.named_modules():
        # mllama decoder layer attribute is "cross_attn"
        if name.endswith("cross_attn"):
            mods.append((name, m))
    return mods

def install_cross_states_leaf_capture(
    model: nn.Module,
    capture_last_n: int,
    storage: List[torch.Tensor],
):
    cross_mods = find_all_cross_attn_modules(model)
    if len(cross_mods) == 0:
        print("[WARN] No cross_attn modules found by name scan.")
        return []

    selected = cross_mods[-capture_last_n:]
    originals: List[Tuple[nn.Module, callable]] = []

    for (mod_name, cross_mod) in selected:
        originals.append((cross_mod, cross_mod.forward))
        orig_forward = cross_mod.forward

        def wrapped_forward(hidden_states, *args, __orig=orig_forward, **kwargs):
            # 1) grab cross_attention_states that is actually used
            cross_states = kwargs.get("cross_attention_states", None)
            if cross_states is None:
                # try common aliases
                for key in ["encoder_hidden_states", "key_value_states", "vision_hidden_states", "image_hidden_states"]:
                    if key in kwargs:
                        cross_states = kwargs[key]
                        break

            # If still none, run original
            if cross_states is None or (not torch.is_tensor(cross_states)) or cross_states.dim() != 3:
                return __orig(hidden_states, *args, **kwargs)

            # 2) cut graph to vision/projector (memory saver) + make leaf so grad guaranteed
            cross_leaf = cross_states.detach().requires_grad_(True)
            cross_leaf.retain_grad()

            # 3) make original forward USE our leaf (ancestor of loss)
            kwargs["cross_attention_states"] = cross_leaf

            # 4) store (keep only last_n modules so list is small)
            storage.append(cross_leaf)

            return __orig(hidden_states, *args, **kwargs)

        cross_mod.forward = wrapped_forward

    print(f"[OK] Patched cross_attn modules: {len(selected)}/{len(cross_mods)} (last_n={capture_last_n})")
    return originals

def cross_states_gradinput_to_patch_heatmap(
    states_list: List[torch.Tensor],
    grid_h: int,
    grid_w: int,
    n_special: int,
    vision_meta: Optional[Dict],
    per_tile_patches: int,
) -> np.ndarray:
    if not states_list:
        raise RuntimeError("No captured cross_attention_states.")

    agg = None
    for s in states_list:
        if s.grad is None:
            continue
        # [B,S,E] -> token score [S]
        x = torch.relu((s * s.grad).sum(dim=-1)).mean(dim=0)  # [S]
        agg = x if agg is None else (agg + x)

    if agg is None:
        raise RuntimeError("No gradients found on captured cross_attention_states (did backward run?)")

    agg = agg.detach().float().cpu()
    S = agg.numel()
    P = grid_h * grid_w

    # same mapping logic you had
    if vision_meta and vision_meta.get("Ntiles", 1) > 1:
        Ntiles = int(vision_meta["Ntiles"])
        T_per_tile = int(vision_meta.get("T", n_special + per_tile_patches))
        patches_all = []
        for t in range(Ntiles):
            off = t * T_per_tile
            st = off + n_special
            ed = st + per_tile_patches
            if ed <= S:
                patches_all.append(agg[st:ed])
        if not patches_all:
            total_needed = Ntiles * P
            if S >= total_needed:
                patches_all = [agg[t * P:(t + 1) * P] for t in range(Ntiles)]
            else:
                raise ValueError(f"Cannot map seq (S={S}) to tiles/patches.")
        v = torch.stack(patches_all, dim=0).sum(dim=0)
    else:
        if S >= n_special + P:
            v = agg[n_special:n_special + P]
        else:
            if S < P:
                raise ValueError(f"Seq too short S={S}, expected >= {P}")
            v = agg[:P]

    v = v / (v.max() + 1e-8)
    return v.numpy().reshape(grid_h, grid_w)


# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--unique_id", required=True, type=str)
    p.add_argument("--out_dir", default="./explain_outputs", type=str)
    p.add_argument("--model_id", default="meta-llama/Llama-3.2-11B-Vision-Instruct", type=str)
    p.add_argument("--checkpoint_dir", default="../trained_models/dn", type=str)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    p.add_argument("--vit_layers", default=1, type=int)
    p.add_argument("--vit_mode", default="last", choices=["last", "rollout"], type=str)
    p.add_argument("--cross_layers", default=4, type=int)
    p.add_argument("--target_class", default=1, type=int)
    p.add_argument("--do_occlusion", action="store_true")
    p.add_argument("--occ_stride", default=2, type=int)
    p.add_argument("--occ_mask_value", default=0.0, type=float)
    p.add_argument("--occ_micro_batch", default=8, type=int)
    p.add_argument("--alpha", default=0.35, type=float)
    p.add_argument("--cmap_pos", default="magma", type=str)
    p.add_argument("--cmap_signed", default="seismic", type=str)
    p.add_argument("--metadata_path", default="../dataset/metadata.json", type=str)
    p.add_argument("--metadata_image_path", default="../dataset/test_summarization/full-test-indent-images.json", type=str)
    p.add_argument("--test_data_path", default="../dataset/test_summarization/total_output.jsonl", type=str)
    p.add_argument("--base_img_dir", default="../saved_images", type=str)
    p.add_argument("--base_rr_dir", default="../physionet.org/files/mimic-cxr/2.1.0/files", type=str)
    p.add_argument("--summary_type", default="plain", type=str)
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from dataclasses import dataclass

    @dataclass
    class ModelArgs:
        model_name_or_path: str = args.model_id
        checkpoint_dir: str = args.checkpoint_dir
        inference: bool = True
        use_cxr_image: bool = True
        use_discharge_note: bool = True
        use_rad_report: bool = False
        use_generated_rad_report: bool = False
        use_pi: bool = False
        summarize: bool = False
        zeroshot: bool = False
        summary_type: str = args.summary_type
        base_img_dir: str = args.base_img_dir
        metadata_path: str = args.metadata_path
        test_data_path: str = args.test_data_path
        test_metadata_image_path: str = args.metadata_image_path

    processor = AutoProcessor.from_pretrained(args.model_id)
    sample = load_sample_by_unique_id(args.unique_id, args, processor)

    model_args = ModelArgs()
    model, _ = load_model(model_args, model_id=args.model_id, inference=True)
    model = model.to(device).eval()

    # build batch
    batch, _messages = build_batch(processor, sample, device)

    # forward for prediction (store scalars BEFORE deleting tensors)
    with torch.inference_mode():
        out = model(**batch)
        logits = out["logits"]  # keep on GPU briefly

        probs_t = torch.softmax(logits.float(), dim=-1)[0]  # [2] on GPU
        pred = int(torch.argmax(probs_t).item())
        p0 = float(probs_t[0].item())
        p1 = float(probs_t[1].item())

        # base_logit for target_class (used when occlusion is off)
        base_logit_pred = float(logits[0, args.target_class].float().item())

    # now safe to delete tensors
    del out, logits, probs_t
    hard_cuda_gc(device)

    base, vision_model = resolve_base_and_vision(model)

    # detect tokens/grid
    pixel_values = batch["pixel_values"]
    aspect_ratio_ids = batch.get("aspect_ratio_ids", None)
    aspect_ratio_mask = batch.get("aspect_ratio_mask", None)

    cls_index, n_special, num_patches, grid_h, grid_w, vision_meta = detect_vision_tokens_and_grid(
        vision_model, pixel_values, processor, aspect_ratio_ids, aspect_ratio_mask
    )

    # -----------------------
    # Stage 1: ViT CLS row
    # -----------------------
    vit_rows = []
    vit_originals = install_vit_cls_row_capture(
        vision_model=vision_model,
        cls_index=cls_index,
        n_special=n_special,
        capture_last_n=args.vit_layers,
        storage=vit_rows,
        debug_once=True,
    )

    with torch.inference_mode():
        kwargs = {"pixel_values": pixel_values}
        if aspect_ratio_ids is not None:
            kwargs["aspect_ratio_ids"] = aspect_ratio_ids
        if aspect_ratio_mask is not None:
            kwargs["aspect_ratio_mask"] = aspect_ratio_mask
        _ = vision_model(**kwargs)

    restore_patched_forwards(vit_originals)
    hm_vit = cls_row_to_heatmap(vit_rows, grid_h, grid_w, mode=args.vit_mode)
    vit_rows.clear()
    
    # vision_model output 삭제
    del kwargs
    if 'out' in locals():
        del out
        
    hard_cuda_gc(device)
    torch.cuda.empty_cache()

    # -----------------------
    # Stage 2: Cross (Grad*Input on cross_states)
    # -----------------------
    cross_states_list: List[torch.Tensor] = []
    cross_originals = install_cross_states_leaf_capture(
        model=model,
        capture_last_n=args.cross_layers,
        storage=cross_states_list,
    )

    model.zero_grad(set_to_none=True)

    dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    with torch.set_grad_enabled(True):
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=dtype):
            out2 = model(**batch)
            target = out2["logits"][:, args.target_class].sum()
        target.backward()

    restore_patched_forwards(cross_originals)

    hm_gradattn = cross_states_gradinput_to_patch_heatmap(
        states_list=cross_states_list,
        grid_h=grid_h,
        grid_w=grid_w,
        n_special=n_special,
        vision_meta=vision_meta,
        per_tile_patches=num_patches,
    )

    # cleanup
    cross_states_list.clear()
    del out2, target
    model.zero_grad(set_to_none=True)
    hard_cuda_gc(device)


    # -----------------------
    # Stage 3: Signed occlusion
    # -----------------------
    if args.do_occlusion:
        hm_occ, base_logit = compute_signed_occlusion_map(
            model=model,
            batch=batch,
            device=device,
            target_class=args.target_class,
            grid_h=grid_h,
            grid_w=grid_w,
            representative_tile=0,
            stride=args.occ_stride,
            mask_value=args.occ_mask_value,
            micro_batch=args.occ_micro_batch,
        )
    else:
        hm_occ = np.zeros((grid_h, grid_w), dtype=np.float32)
        base_logit = base_logit_pred

    # -----------------------
    # Save
    # -----------------------
    meta_title = f"{args.unique_id} | pred={pred} p1={p1:.3f} base_logit={base_logit:.3f}"
    fig_path = out_dir / f"{args.unique_id}_4panel.png"

    save_4panel(
        out_path=str(fig_path),
        image=sample["image"],
        hm_vit=hm_vit,
        hm_gradattn=hm_gradattn,
        hm_occ_signed=hm_occ,
        meta_title=meta_title,
        alpha=args.alpha,
        cmap_pos=args.cmap_pos,
        cmap_signed=args.cmap_signed,
    )

    summary = {
        "unique_id": args.unique_id,
        "discharge_note": sample.get("discharge_note"),
        "radiology_report": sample.get("radiology_report"),
        "generated_radiology_report": sample.get("generated_radiology_report"),
        "image_path": sample.get("image_path"),
        "label": sample.get("label"),
        "prob_class_1": p1,
    }

    summary_path = out_dir / f"{args.unique_id}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[OK] saved: {fig_path}")

if __name__ == "__main__":
    main()