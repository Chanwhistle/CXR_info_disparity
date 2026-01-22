#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import gc
import argparse
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List, Sequence, Any
import math

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor
from scipy import ndimage
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from dataloader import load_hash2meta_dict, CXRDecisionTree
from model import load_model


# ============================================================================
# Configuration
# ============================================================================
@dataclass
class RunConfig:
    alpha: float = 0.5
    cmap: str = "jet"
    interp: str = "bilinear"
    blur_radius: float = 2.0
    patch_sigma: float = 1.5
    normalize_gamma: float = 0.7


# ============================================================================
# Utilities
# ============================================================================
def hard_cuda_gc(device: torch.device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _pil_resample(interp: str) -> int:
    interp = (interp or "bilinear").lower().strip()
    if interp == "bicubic":
        return Image.BICUBIC
    return Image.BILINEAR


def normalize_map(
    x: np.ndarray,
    clip_low: float = 50.0,
    clip_high: float = 99.5,
    gamma: float = 1.5,
    stat_mask: Optional[np.ndarray] = None,
    smooth_sigma: float = 0.5,
) -> np.ndarray:
    x = x.astype(np.float32)
    finite = np.isfinite(x)
    if stat_mask is None:
        m = finite
    else:
        m = finite & stat_mask.astype(bool)

    if m.sum() == 0:
        return np.zeros_like(x, dtype=np.float32)

    v = x[m]
    lo, hi = np.percentile(v, clip_low), np.percentile(v, clip_high)
    if hi <= lo:
        lo, hi = float(v.min()), float(v.max() + 1e-8)

    y = np.clip(x, lo, hi)
    y = (y - lo) / (hi - lo + 1e-8)
    y = np.power(y, gamma)

    if smooth_sigma > 0:
        y = ndimage.gaussian_filter(y, sigma=smooth_sigma, mode="nearest")

    return y.astype(np.float32)


def tile_index_to_rc(tile_idx: int, rows: int, cols: int, order: str = "col-major"):
    if order == "row-major":
        rr = tile_idx // cols
        cc = tile_idx % cols
    else:  # "col-major"
        rr = tile_idx % rows
        cc = tile_idx // rows
    return rr, cc

def _get_llm_layers(base_model: nn.Module):
    # Mllama / LLaMA-Vision 계열에서 자주 나오는 경로들까지 포함
    candidate_paths = [
        # 일반 LlamaForCausalLM 계열
        "model.layers",
        "layers",

        # base_model 안에 language_model이 있는 멀티모달 래퍼
        "language_model.model.layers",
        "language_model.layers",

        # 혹시 model이 한 단계 더 감싸진 케이스
        "base_model.model.layers",
        "base_model.layers",
        "base_model.language_model.model.layers",
        "base_model.language_model.layers",
    ]

    for path in candidate_paths:
        layers = _get_nested_attr(base_model, path)
        if isinstance(layers, (list, nn.ModuleList)) and len(layers) > 0:
            return layers

    # ---- fallback: 전체 모듈을 훑어서 'layers'라는 ModuleList를 찾아보기 ----
    for name, mod in base_model.named_modules():
        if name.endswith("layers") and isinstance(mod, nn.ModuleList) and len(mod) > 0:
            # 디코더 레이어처럼 보이는지 간단히 확인 (self_attn이 있거나 q_proj/k_proj가 있는지)
            layer0 = mod[0]
            if hasattr(layer0, "self_attn") or any(hasattr(layer0, x) for x in ["attention", "attn"]):
                return mod

    raise AttributeError(
        "Cannot find LLM decoder layers. "
        "Tried paths: " + ", ".join(candidate_paths)
    )

def _find_cross_attn_modules_in_layers(layers: nn.ModuleList) -> List[Tuple[int, str, nn.Module]]:
    """
    decoder layer들에서 cross-attn 모듈을 찾아서 (layer_idx, name, module) 리스트로 반환
    Mllama는 보통 layer.cross_attn 가 있음.
    """
    found = []
    for li, layer in enumerate(layers):
        # 가장 흔한 이름
        for name in ["cross_attn", "cross_attention", "encoder_attn"]:
            if hasattr(layer, name):
                m = getattr(layer, name)
                if isinstance(m, nn.Module):
                    found.append((li, name, m))
                    break
        else:
            # 혹시 레이어 내부에 다른 이름으로 들어간 경우 named_modules로 스캔
            for n, m in layer.named_modules():
                nl = n.lower()
                if ("cross" in nl and "attn" in nl) or ("encoder" in nl and "attn" in nl):
                    # q_proj/k_proj 있는 attention 류만
                    if hasattr(m, "q_proj") and hasattr(m, "k_proj"):
                        found.append((li, n, m))
                        break
    return found

def _extract_cross_states(args, kwargs) -> Optional[torch.Tensor]:
    """
    cross-attn의 encoder side hidden states (image tokens)를 kwargs/args에서 추출
    """
    for key in ["encoder_hidden_states", "cross_attention_states", "key_value_states",
                "vision_hidden_states", "image_hidden_states"]:
        v = kwargs.get(key, None)
        if torch.is_tensor(v) and v.dim() == 3:
            return v
    # positional에도 있을 수 있음
    for a in args:
        if torch.is_tensor(a) and a.dim() == 3:
            return a
    return None

@torch.no_grad()
def capture_llm_self_attn_lasttok_to_image_probs(
    base_model: nn.Module,
    batch: Dict[str, torch.Tensor],
    capture_last_n: int = 1,
):
    """
    (이름은 그대로 두되) 실제로는 cross-attn 기반으로
    last text token이 image tokens(encoder states)에 주는 attention probs를 뽑아옴.

    Return: vec (S_img,) averaged across selected decoder layers and heads.
    """

    # ---- 0) last non-pad text token index ----
    attn_mask = batch.get("attention_mask", None)
    if attn_mask is None:
        last_idx = -1
    else:
        last_idx = int(attn_mask[0].long().sum().item() - 1)
        last_idx = max(last_idx, 0)

    # ---- 1) decoder layers 가져오기 ----
    layers = _get_llm_layers(base_model)  # 너가 이미 고쳐둔 함수
    if layers is None or len(layers) == 0:
        return None

    # ---- 2) cross-attn 모듈 찾기 ----
    cross_mods = _find_cross_attn_modules_in_layers(layers)
    if len(cross_mods) == 0:
        # 이 경우는 모델이 아예 cross-attn을 안 쓰거나, 모듈명이 완전 다를 수 있음
        return None

    # 마지막 N개 레이어만
    cross_mods = cross_mods[-capture_last_n:] if capture_last_n > 0 else cross_mods

    collected: List[torch.Tensor] = []
    originals: List[Tuple[nn.Module, Any]] = []

    def make_wrapper(cross_mod: nn.Module, orig_forward):
        q_proj = getattr(cross_mod, "q_proj", None)
        k_proj = getattr(cross_mod, "k_proj", None)
        num_heads = getattr(cross_mod, "num_heads", getattr(cross_mod, "num_attention_heads", None))

        def wrapped(hidden_states, *args, **kwargs):
            # 먼저 원래 forward 실행
            out = orig_forward(hidden_states, *args, **kwargs)

            # q/k 프로젝션 없는 구현이면 포기
            if q_proj is None or k_proj is None:
                return out
            if (not torch.is_tensor(hidden_states)) or hidden_states.dim() != 3:
                return out

            # encoder_hidden_states (이미지 토큰) 추출
            cross_states = _extract_cross_states(args, kwargs)
            if cross_states is None:
                return out

            # shapes
            B, Ttxt, E = hidden_states.shape
            Simg = cross_states.shape[1]

            # 헤드 수 / head dim 추정
            H = int(num_heads) if num_heads is not None else max(1, E // 128)
            d = E // H
            if d * H != E:
                return out

            # last text token index clamp
            tidx = min(max(last_idx, 0), Ttxt - 1)

            # 변경 (메모리 크게 절약)
            q = q_proj(hidden_states[:, tidx:tidx+1, :])          # dtype 유지 (bf16)
            k = k_proj(cross_states)                               # dtype 유지 (bf16)

            q = q.view(B, 1, H, d).transpose(1, 2)                 # [B,H,1,d]
            k = k.view(B, Simg, H, d).transpose(1, 2)              # [B,H,S,d]

            scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(d))  # [B,H,1,S] (bf16)
            probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)[:, :, 0, :]  # softmax만 fp32

            # ✅ GPU에 쌓지 말고 바로 CPU로
            collected.append(probs.detach().cpu())
            return out

        return wrapped

    # ---- 3) hook install ----
    for (li, name, mod) in cross_mods:
        originals.append((mod, mod.forward))
        mod.forward = make_wrapper(mod, mod.forward)

    # ---- 4) forward once ----
    try:
        _ = base_model(**batch, use_cache=False)
    finally:
        for mod, f in originals:
            mod.forward = f

    if len(collected) == 0:
        return None

    x = torch.stack(collected, dim=0).mean(dim=0)  # collected가 이미 CPU
    x = x.mean(dim=1).mean(dim=0)                  # [S_img]
    
    return x.float().cpu()




def capture_llm_self_attn_lasttok_to_image_gradcam(
    model: nn.Module,
    base_model: nn.Module,
    batch: Dict[str, torch.Tensor],
    target_class: int,
    capture_last_n: int = 1,
) -> Optional[torch.Tensor]:
    """
    Memory-safe Grad×Attn for *cross-attn* (last text token -> image encoder states).
    - DOES NOT use output_attentions=True (so SDPA stays on, avoids full [B,H,Q,K] materialization)
    - Captures only last-token q and full image-side k,v for selected layers.
    - After backward, reconstructs dL/d(scores_last) using grad of last-token attn output.

    Returns: vec (S_img,) CPU float32
    """

    # ---- 0) last non-pad text token index ----
    attn_mask = batch.get("attention_mask", None)
    if attn_mask is None:
        last_idx = int(batch["input_ids"].shape[1] - 1)
    else:
        last_idx = int(attn_mask[0].long().sum().item() - 1)
        last_idx = max(last_idx, 0)

    # ---- 1) decoder layers / cross-attn modules ----
    layers = _get_llm_layers(base_model)
    if not layers:
        return None

    cross_mods = _find_cross_attn_modules_in_layers(layers)
    if not cross_mods:
        return None

    cross_mods = cross_mods[-capture_last_n:] if capture_last_n > 0 else cross_mods

    stor: List[Dict[str, Any]] = []
    originals: List[Tuple[nn.Module, Any]] = []

    # ---- 2) wrap cross-attn forward (NO output_attentions) ----
    for (li, name, cross_mod) in cross_mods:
        q_proj = getattr(cross_mod, "q_proj", None)
        k_proj = getattr(cross_mod, "k_proj", None)
        v_proj = getattr(cross_mod, "v_proj", None)
        o_proj = getattr(cross_mod, "o_proj", None)

        num_heads = getattr(cross_mod, "num_heads", getattr(cross_mod, "num_attention_heads", None))
        head_dim  = getattr(cross_mod, "head_dim", None)

        if q_proj is None or k_proj is None or v_proj is None or o_proj is None:
            continue

        orig_forward = cross_mod.forward
        originals.append((cross_mod, orig_forward))

        def make_wrapped(orig_fwd, _li=li, _name=name):
            def wrapped(hidden_states, *args, **kwargs):
                # hidden_states: [B, Ttxt, E]
                if (not torch.is_tensor(hidden_states)) or hidden_states.dim() != 3:
                    return orig_fwd(hidden_states, *args, **kwargs)

                cross_states = _extract_cross_states(args, kwargs)  # [B,Simg,E] (image side)
                if cross_states is None:
                    return orig_fwd(hidden_states, *args, **kwargs)

                B, Ttxt, E = hidden_states.shape
                Simg = int(cross_states.shape[1])

                H = int(num_heads) if num_heads is not None else max(1, E // 128)
                d = int(head_dim) if head_dim is not None else (E // H)
                if d * H != E:
                    return orig_fwd(hidden_states, *args, **kwargs)

                tidx = min(max(last_idx, 0), Ttxt - 1)

                # ---- capture q_last, k_img, v_img in head space (bf16 유지) ----
                # q: [B,1,E] -> [B,H,1,d]
                q = q_proj(hidden_states[:, tidx:tidx+1, :])
                q = q.view(B, 1, H, d).transpose(1, 2)  # [B,H,1,d]

                # k,v: [B,Simg,E] -> [B,H,Simg,d]
                k = k_proj(cross_states).view(B, Simg, H, d).transpose(1, 2)  # [B,H,Simg,d]
                v = v_proj(cross_states).view(B, Simg, H, d).transpose(1, 2)  # [B,H,Simg,d]

                # ---- run original forward (SDPA path 유지, 메모리 절약) ----
                out = orig_fwd(hidden_states, *args, **kwargs)

                # out[0] should be attn_output (B,Ttxt,E) for many implementations
                attn_out = None
                if isinstance(out, (tuple, list)) and len(out) >= 1 and torch.is_tensor(out[0]):
                    attn_out = out[0]
                elif torch.is_tensor(out):
                    attn_out = out

                # we need gradient of last token attn_output
                if attn_out is not None and attn_out.requires_grad:
                    attn_out.retain_grad()

                stor.append({
                    "layer": _li,
                    "name": _name,
                    "q": q,            # [B,H,1,d]
                    "k": k,            # [B,H,S,d]
                    "v": v,            # [B,H,S,d]
                    "o_proj": o_proj,  # module ref
                    "tidx": tidx,
                    "attn_out": attn_out,  # [B,T,E]
                })
                return out
            return wrapped

        cross_mod.forward = make_wrapped(orig_forward)

    if not originals:
        return None

    # ---- 3) forward + backward ----
    model.zero_grad(set_to_none=True)
    out = model(**batch, use_cache=False)
    logit = out["logits"][0, int(target_class)]
    logit.backward()

    # ---- 4) restore ----
    for m, f in originals:
        m.forward = f

    if not stor:
        model.zero_grad(set_to_none=True)
        return None

    # ---- 5) reconstruct grad wrt scores_last WITHOUT full [Q,K] ----
    cams: List[torch.Tensor] = []

    for item in stor:
        attn_out = item["attn_out"]
        if attn_out is None or attn_out.grad is None:
            continue

        B = item["q"].shape[0]
        H = item["q"].shape[1]
        S = item["k"].shape[2]
        d = item["q"].shape[-1]
        tidx = item["tidx"]

        # grad of attn output at last token (after o_proj output, shape [B,E])
        grad_out_last = attn_out.grad[:, tidx, :].detach()  # [B,E]

        # build last-token attention subgraph: scores -> softmax -> out -> o_proj
        q = item["q"].detach()  # treat q,k,v constants for reconstruction
        k = item["k"].detach()
        v = item["v"].detach()
        o_proj = item["o_proj"]

        # scores_last: [B,H,1,S] (require grad)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
        scores = scores.float()  # softmax 안정성
        scores.requires_grad_(True)

        probs = torch.softmax(scores, dim=-1)  # [B,H,1,S]
        out_h = torch.matmul(probs, v.float())  # [B,H,1,d]  (v는 float로 맞춤)

        # merge heads -> [B,1,E]
        out_merge = out_h.transpose(1, 2).contiguous().view(B, 1, H * d)  # [B,1,E]

        # apply o_proj -> [B,1,E]
        out_proj = o_proj(out_merge.to(dtype=o_proj.weight.dtype))

        # grad wrt scores
        # grad_outputs: [B,1,E]
        grad_outputs = grad_out_last.view(B, 1, -1).to(dtype=out_proj.dtype)

        grad_scores = torch.autograd.grad(
            outputs=out_proj,
            inputs=scores,
            grad_outputs=grad_outputs,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )[0]

        if grad_scores is None:
            continue

        # attribution: ReLU(probs * grad_scores) on last query
        cam = torch.relu(probs.detach().cpu()[:, :, 0, :] * grad_scores.detach().cpu()[:, :, 0, :])  # [B,H,S] CPU
        cams.append(cam)

    model.zero_grad(set_to_none=True)

    if not cams:
        return None

    x = torch.stack(cams, dim=0).mean(dim=0)  # [B,H,S]
    x = x.mean(dim=1).mean(dim=0)             # [S]
    return x.float().cpu()




# ============================================================================
# helpers: find modules / nested attrs
# ============================================================================
def _get_nested_attr(obj, path: str):
    cur = obj
    for p in path.split("."):
        if not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur


def get_vision_layers(vision_model: nn.Module):
    for path in ["transformer.layers", "layers", "encoder.layers", "encoder.layer", "model.layers"]:
        layers = _get_nested_attr(vision_model, path)
        if isinstance(layers, (list, nn.ModuleList)) and len(layers) > 0:
            return layers
    raise AttributeError("Cannot find vision transformer layers.")


def get_self_attn_module(layer: nn.Module) -> Optional[nn.Module]:
    for name in ["self_attn", "attn", "attention", "self_attention"]:
        if hasattr(layer, name):
            return getattr(layer, name)
    # fallback scan
    for n, m in layer.named_modules():
        ln = n.lower()
        if ("attn" in ln or "attention" in ln) and hasattr(m, "forward"):
            return m
    return None


# ============================================================================
# Token layout helpers (IMPORTANT: head offset must be applied)
# ============================================================================
def detect_token_layout_from_S(S_per_tile: int, grid_h: int, grid_w: int):
    P = grid_h * grid_w

    # PATCH-ONLY (no CLS/head)
    if S_per_tile == P:
        return 0, P, 0

    # common (CLS + patches + tail)
    if S_per_tile == 1 + P + 7:
        return 1, P, 7
    if S_per_tile == 1 + P:
        return 1, P, 0
    if S_per_tile >= 1 + P:
        tail = S_per_tile - 1 - P
        return 1, P, tail

    raise ValueError(f"S_per_tile too short: {S_per_tile} < {P}")


# ============================================================================
# Data Loading
# ============================================================================
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

    rad_report = None
    generated_rad_report = None
    if selected_img_data_path:
        path_parts = selected_img_data_path.split("/")[:3]
        if len(path_parts) == 3:
            rr_relative_path = "/".join(path_parts) + ".txt"
            original_report_path = os.path.join(args.base_rr_dir, rr_relative_path)

            report_dir = "/".join(path_parts[:-1])
            report_filename = path_parts[-1] + ".txt"
            generated_rr_relative_path = f"{report_dir}/generated_{report_filename}"
            generated_report_path = os.path.join(args.base_rr_dir, generated_rr_relative_path)

            if os.path.exists(original_report_path):
                with open(original_report_path, "r", encoding="utf-8") as f:
                    rad_report = f.read().replace("\n", " ").strip()

            if os.path.exists(generated_report_path):
                with open(generated_report_path, "r", encoding="utf-8") as f:
                    generated_rad_report = f.read().replace("\n", " ").strip()

    return {
        "id": unique_id,
        "discharge_note": discharge_note,
        "radiology_report": rad_report,
        "generated_radiology_report": generated_rad_report,
        "label": label,
        "image": img,
        "image_path": real_path,
    }


# ============================================================================
# Batch build (NO forced truncation)
# ============================================================================
def build_batch(processor: AutoProcessor, sample: Dict, device: torch.device) -> Dict:
    system_prompt = (
        "A clinical document and a single, most recent chest X-ray (CXR) image from the patient are provided. "
        "Based on the clinical context and the provided CXR image, assess how likely the patient's "
        "out-of-hospital mortality is within 30 days."
    )

    if sample.get("discharge_note"):
        user_prompt = (
            f"Here is the clinical document:\n{sample['discharge_note']}\n\n"
            "Based on the provided clinical information and single CXR image, how likely is the patient's "
            "out-of-hospital mortality within 30 days? "
            "Please do not explain the reason and respond with one word only: 0:alive, 1:death.\n\nAssistant:"
        )
    else:
        user_prompt = (
            "Based on the provided single CXR image, how likely is the patient's out-of-hospital mortality "
            "within 30 days? Please do not explain the reason and respond with one word only: 0:alive, 1:death.\n\nAssistant:"
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False)

    processed = processor(
        text=[text],
        images=[[sample["image"]]],
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in processed.items()}


# ============================================================================
# Layout / Grid
# ============================================================================
def infer_tile_layout(ntiles: int) -> Tuple[int, int]:
    if ntiles == 4:
        return 2, 2
    if ntiles == 1:
        return 1, 1
    if ntiles == 2:
        return 1, 2
    side = int(round(ntiles ** 0.5))
    if side * side == ntiles:
        return side, side
    return 1, ntiles


def infer_grid_from_batch(batch: Dict, model: nn.Module) -> Tuple[int, int, int, int, int, int]:
    pv = batch["pixel_values"]
    base_model = getattr(model, "base_model", model)
    vision_model = getattr(base_model, "vision_model", None)
    config = getattr(vision_model, "config", None)

    patch_size = getattr(config, "patch_size", 14)
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[0]

    if pv.dim() == 6:
        ntiles = int(pv.shape[2])
        tile_h, tile_w = int(pv.shape[-2]), int(pv.shape[-1])
    elif pv.dim() == 5:
        ntiles = int(pv.shape[1])
        tile_h, tile_w = int(pv.shape[-2]), int(pv.shape[-1])
    else:
        ntiles = 1
        tile_h, tile_w = int(pv.shape[-2]), int(pv.shape[-1])

    grid_h = tile_h // patch_size
    grid_w = tile_w // patch_size
    return int(patch_size), int(grid_h), int(grid_w), int(ntiles), int(tile_h), int(tile_w)


def get_valid_tile_indices(batch: Dict, ntiles: int) -> List[int]:
    arm = batch.get("aspect_ratio_mask", None)
    if arm is None:
        return list(range(ntiles))

    if arm.dim() == 3:
        mask = arm[0, 0].detach().cpu().bool().numpy()
    elif arm.dim() == 2:
        mask = arm[0].detach().cpu().bool().numpy()
    else:
        return list(range(ntiles))

    return [i for i in range(min(len(mask), ntiles)) if mask[i]]


def detect_padding_patches(
    pv: torch.Tensor,
    ntiles: int,
    grid_h: int,
    grid_w: int,
    patch_size: int,
    eps_fill: float = 1e-3,
    eps_std: float = 1e-4,
    border_frac: float = 0.08,
) -> np.ndarray:
    """
    padding을 '특정 상수(fill value)'로 채워 넣은 경우를 강하게 잡아내는 버전.
    - tile 테두리에서 fill 값(채널별)을 추정
    - patch가 fill 값과 거의 같거나 내부 변화가 거의 없으면 padding(False)
    """
    if pv.dim() == 6:        # [B, Nimg, Ntiles, C, H, W]
        tiles = pv[0, 0].detach().cpu().numpy()
    elif pv.dim() == 5:      # [B, Ntiles, C, H, W]
        tiles = pv[0].detach().cpu().numpy()
    elif pv.dim() == 4:      # [B, C, H, W]
        tiles = pv.detach().cpu().numpy()[0:1]
    else:
        raise ValueError(f"Unsupported pixel_values shape: {tuple(pv.shape)}")

    ntiles_actual, C, tile_h, tile_w = tiles.shape
    ntiles_use = min(ntiles_actual, ntiles)

    valid = np.ones((ntiles_use, grid_h, grid_w), dtype=bool)

    def estimate_fill_value_per_channel(tile_chw: np.ndarray) -> np.ndarray:
        H, W = tile_chw.shape[1], tile_chw.shape[2]
        bw = max(1, int(round(W * border_frac)))
        bh = max(1, int(round(H * border_frac)))

        top = tile_chw[:, :bh, :]
        bottom = tile_chw[:, H-bh:, :]
        left = tile_chw[:, :, :bw]
        right = tile_chw[:, :, W-bw:]

        border = np.concatenate(
            [top.reshape(C, -1), bottom.reshape(C, -1), left.reshape(C, -1), right.reshape(C, -1)],
            axis=1
        )

        q = np.round(border / eps_fill) * eps_fill
        fill = []
        for ch in range(C):
            vals, counts = np.unique(q[ch], return_counts=True)
            fill.append(vals[np.argmax(counts)])
        return np.array(fill, dtype=np.float32)

    for t in range(ntiles_use):
        tile = tiles[t].astype(np.float32)
        fill_c = estimate_fill_value_per_channel(tile)

        for r in range(grid_h):
            r0, r1 = r * patch_size, min((r + 1) * patch_size, tile_h)
            for c in range(grid_w):
                c0, c1 = c * patch_size, min((c + 1) * patch_size, tile_w)
                patch = tile[:, r0:r1, c0:c1]

                diff = np.max(np.abs(patch - fill_c[:, None, None]))
                if diff < eps_fill:
                    valid[t, r, c] = False
                    continue

                std_score = float(patch.reshape(C, -1).std(axis=1).mean())
                if std_score < eps_std:
                    valid[t, r, c] = False

    if ntiles_use < ntiles:
        pad = np.zeros((ntiles - ntiles_use, grid_h, grid_w), dtype=bool)
        valid = np.concatenate([valid, pad], axis=0)

    return valid


# ============================================================================
# Pixel-canvas scatter mapping
# ============================================================================
def _tensor_to_1d_np(x: torch.Tensor) -> np.ndarray:
    x = x.detach().to(dtype=torch.float32).cpu()
    if x.dim() >= 1:
        x = x[0]
    if x.dim() > 1:
        x = x.flatten()
    return x.numpy().astype(np.float32)


def scalar_to_tile_patchvals_packed(
    scalars: torch.Tensor,
    grid_h: int,
    grid_w: int,
    ntiles: int,
    tile_indices: List[int],
    patch_mask: Optional[np.ndarray] = None,
) -> List[np.ndarray]:
    """
    scalars: [B,S] or [S] or anything flattenable. We map patch tokens ONLY.
    IMPORTANT: apply head offset (CLS etc).
    """
    s0 = _tensor_to_1d_np(scalars)
    N = int(s0.shape[0])

    if ntiles > 1 and (N % ntiles == 0):
        ntiles_eff = ntiles
        S_per = N // ntiles
    else:
        ntiles_eff = 1
        S_per = N
        
    head, P, _tail = detect_token_layout_from_S(S_per, grid_h, grid_w)

    out: List[np.ndarray] = []
    for tile_idx in tile_indices:
        if ntiles_eff == 1:
            base = 0
            if tile_idx != 0:
                out.append(np.zeros((P,), dtype=np.float32))
                continue
        else:
            if tile_idx >= ntiles_eff:
                out.append(np.zeros((P,), dtype=np.float32))
                continue
            base = tile_idx * S_per

        st = base + head
        ed = min(st + P, base + S_per)
        vals = np.zeros((P,), dtype=np.float32)
        chunk = s0[st:ed]
        vals[: chunk.shape[0]] = chunk

        if patch_mask is not None and tile_idx < patch_mask.shape[0]:
            vals *= patch_mask[tile_idx].reshape(-1).astype(np.float32)

        out.append(vals.astype(np.float32))

    return out


def smooth_patchvals_packed(
    patchvals_packed: List[np.ndarray],
    grid_h: int,
    grid_w: int,
    sigma: float = 1.5,
) -> List[np.ndarray]:
    smoothed = []
    for vals in patchvals_packed:
        grid = vals.reshape(grid_h, grid_w)
        smoothed_grid = ndimage.gaussian_filter(grid, sigma=sigma, mode="nearest")
        smoothed.append(smoothed_grid.reshape(-1).astype(np.float32))
    return smoothed


def tile_patchvals_to_mosaic_pixel_packed(
    patchvals_packed: List[np.ndarray],
    tile_indices: List[int],
    ntiles: int,
    grid_h: int,
    grid_w: int,
    patch_size: int,
    tile_h: int,
    tile_w: int,
) -> np.ndarray:
    rows, cols = infer_tile_layout(ntiles)
    mosaic_h = rows * tile_h
    mosaic_w = cols * tile_w
    heat = np.zeros((mosaic_h, mosaic_w), dtype=np.float32)

    P = grid_h * grid_w
    if len(patchvals_packed) != len(tile_indices):
        raise ValueError("patchvals_packed and tile_indices length mismatch")

    for i, tile_idx in enumerate(tile_indices):
        if tile_idx >= ntiles:
            continue
        vals = patchvals_packed[i]
        rr, cc = tile_index_to_rc(tile_idx, rows, cols, order="col-major")
        y_tile0 = rr * tile_h
        x_tile0 = cc * tile_w

        for p in range(P):
            r = p // grid_w
            c = p % grid_w
            y0 = y_tile0 + r * patch_size
            x0 = x_tile0 + c * patch_size
            y1 = min(y0 + patch_size, y_tile0 + tile_h)
            x1 = min(x0 + patch_size, x_tile0 + tile_w)
            heat[y0:y1, x0:x1] = float(vals[p])

    return heat


def patchmask_to_mosaic_pixel_mask(
    patch_mask: np.ndarray,
    tile_indices: List[int],
    ntiles: int,
    grid_h: int,
    grid_w: int,
    patch_size: int,
    tile_h: int,
    tile_w: int,
) -> np.ndarray:
    rows, cols = infer_tile_layout(ntiles)
    mosaic_h = rows * tile_h
    mosaic_w = cols * tile_w
    m = np.zeros((mosaic_h, mosaic_w), dtype=bool)

    P = grid_h * grid_w
    for tile_idx in tile_indices:
        if tile_idx >= ntiles or tile_idx >= patch_mask.shape[0]:
            continue
        if not patch_mask[tile_idx].any():
            continue

        rr, cc = tile_index_to_rc(tile_idx, rows, cols, order="col-major")
        y_tile0 = rr * tile_h
        x_tile0 = cc * tile_w

        pm = patch_mask[tile_idx].reshape(-1)
        for p in range(P):
            if not pm[p]:
                continue
            r = p // grid_w
            c = p % grid_w
            y0 = y_tile0 + r * patch_size
            x0 = x_tile0 + c * patch_size
            y1 = min(y0 + patch_size, y_tile0 + tile_h)
            x1 = min(x0 + patch_size, x_tile0 + tile_w)
            m[y0:y1, x0:x1] = True

    return m


def mosaic_pixel_to_original(
    heat_mosaic_px: np.ndarray,
    original_img: Image.Image,
    interp: str = "bilinear",
    blur_radius_px: float = 0.0,
    content_mask_px: Optional[np.ndarray] = None,
    crop_to_content: bool = True,
) -> np.ndarray:
    heat = heat_mosaic_px.astype(np.float32)

    if crop_to_content and content_mask_px is not None and content_mask_px.any():
        ys, xs = np.where(content_mask_px)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        heat = heat[y0:y1, x0:x1]

    if blur_radius_px and blur_radius_px > 0:
        heat = ndimage.gaussian_filter(heat, sigma=blur_radius_px, mode="nearest")

    heat = np.clip(heat, 0, 1)

    orig_w, orig_h = original_img.size
    zoom_factors = (orig_h / heat.shape[0], orig_w / heat.shape[1])
    heat_resized = ndimage.zoom(heat, zoom_factors, order=3)
    return np.clip(heat_resized, 0, 1).astype(np.float32)


# ============================================================================
# Hooks / Captures
# ============================================================================
class TensorCapture:
    def __init__(self):
        self.tensor: Optional[torch.Tensor] = None

    def hook(self, module, inputs, output):
        t = self._extract(output)
        if t is None or (not torch.is_tensor(t)):
            return
        self.tensor = t

    @staticmethod
    def _extract(output):
        if torch.is_tensor(output):
            return output
        if isinstance(output, (tuple, list)) and len(output) > 0 and torch.is_tensor(output[0]):
            return output[0]
        if isinstance(output, dict):
            for k in ["last_hidden_state", "hidden_states", "output"]:
                v = output.get(k, None)
                if torch.is_tensor(v):
                    return v
        return None


def install_capture(module: nn.Module) -> Tuple[TensorCapture, torch.utils.hooks.RemovableHandle]:
    cap = TensorCapture()
    h = module.register_forward_hook(cap.hook)
    return cap, h


def get_vision_layers_list(vision_model: nn.Module):
    for path in [
        "layers",
        "encoder.layers",
        "encoder.layer",
        "model.layers",
        "vision_encoder.layers",
        "transformer.layers",
        "global_transformer.layers",
    ]:
        layers = _get_nested_attr(vision_model, path)
        if layers is not None and isinstance(layers, (list, nn.ModuleList)):
            return layers
    return None


def get_vision_layer_module(vision_model: nn.Module, layer_idx: int) -> nn.Module:
    layers = get_vision_layers_list(vision_model)
    if layers is None:
        raise AttributeError("Could not locate vision layers list")
    if not (0 <= layer_idx < len(layers)):
        raise IndexError(f"Vision layer_idx out of range: {layer_idx} / {len(layers)}")
    return layers[layer_idx]


# ============================================================================
# Cross-attention capture (last non-pad token only)
# ============================================================================
def find_cross_attn_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    mods = []
    for name, mod in model.named_modules():
        nl = name.lower()
        if ("cross_attn" in nl) or ("cross_attention" in nl) or (nl.endswith("crossattn")):
            if hasattr(mod, "q_proj") and hasattr(mod, "k_proj"):
                mods.append((name, mod))
    return mods


def _extract_cross_states(args, kwargs):
    for key in [
        "encoder_hidden_states",
        "cross_attention_states",
        "key_value_states",
        "vision_hidden_states",
        "image_hidden_states",
    ]:
        v = kwargs.get(key, None)
        if torch.is_tensor(v) and v.dim() == 3:
            return v
    for a in args[1:]:
        if torch.is_tensor(a) and a.dim() == 3:
            return a
    return None


def capture_cross_attn_lasttok_probs(
    base_model: nn.Module,
    batch: Dict[str, torch.Tensor],
    capture_last_n: int,
) -> Optional[torch.Tensor]:
    """
    Return: vec (Simg,) averaged across selected cross-attn modules and heads.
    Uses last non-pad token index from batch['attention_mask'].
    """
    attn_mask = batch.get("attention_mask", None)
    if attn_mask is None:
        last_idx = -1
    else:
        last_idx = int(attn_mask[0].long().sum().item() - 1)
        last_idx = max(last_idx, 0)

    cross_mods = find_cross_attn_modules(base_model)
    if len(cross_mods) == 0:
        return None

    selected = cross_mods[-capture_last_n:] if capture_last_n > 0 else cross_mods

    collected: List[torch.Tensor] = []
    originals: List[Tuple[nn.Module, Any]] = []

    def make_wrapper(cross_mod: nn.Module, orig_forward):
        q_proj = getattr(cross_mod, "q_proj", None)
        k_proj = getattr(cross_mod, "k_proj", None)
        num_heads = getattr(cross_mod, "num_heads", getattr(cross_mod, "num_attention_heads", None))

        def wrapped_forward(hidden_states, *args, **kwargs):
            out = orig_forward(hidden_states, *args, **kwargs)

            if q_proj is None or k_proj is None:
                return out
            if (not torch.is_tensor(hidden_states)) or hidden_states.dim() != 3:
                return out

            cross_states = _extract_cross_states(args, kwargs)
            if cross_states is None:
                return out

            try:
                B, Ttxt, E = hidden_states.shape
                Simg = cross_states.shape[1]
                H = int(num_heads) if num_heads is not None else max(1, E // 64)
                d = E // H
                if d * H != E:
                    return out

                tidx = min(max(last_idx, 0), Ttxt - 1)

                q = q_proj(hidden_states[:, tidx:tidx + 1, :])  # (B,1,E)
                k = k_proj(cross_states)                       # (B,Simg,E)

                q = q.view(B, 1, H, d).transpose(1, 2)         # (B,H,1,d)
                k = k.view(B, Simg, H, d).transpose(1, 2)      # (B,H,Simg,d)

                scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
                scores = scores * (1.0 / math.sqrt(d))
                probs = torch.softmax(scores, dim=-1)[:, :, 0, :]              # (B,H,Simg)

                collected.append(probs.detach())

            except Exception:
                pass

            return out

        return wrapped_forward

    for _, mod in selected:
        originals.append((mod, mod.forward))
        mod.forward = make_wrapper(mod, mod.forward)

    with torch.no_grad():
        _ = base_model(**batch, use_cache=False)

    for mod, orig in originals:
        mod.forward = orig

    if len(collected) == 0:
        return None

    x = torch.stack(collected, dim=0).mean(dim=0)  # (B,H,Simg)
    x = x.mean(dim=1)[0]                           # (Simg,)
    return x


# ============================================================================
# Projector token flatten
# ============================================================================
def flatten_proj_tokens(x: torch.Tensor) -> torch.Tensor:
    """
    projector output shape might be:
      [B, S, D] or [B, Ntiles, S, D] or [B, Nimg, Ntiles, S, D]
    return: [B, S_total, D]
    """
    if x.dim() == 3:
        return x
    if x.dim() == 4:
        B, N, S, D = x.shape
        return x.reshape(B, N * S, D)
    if x.dim() == 5:
        B, Nimg, Ntiles, S, D = x.shape
        return x.reshape(B, Nimg * Ntiles * S, D)
    return x.view(x.shape[0], -1, x.shape[-1])


# ============================================================================
# Heatmap overlay helpers
# ============================================================================
def normalize_signed_map(hm: np.ndarray, clip_abs_percentile: float = 99.0) -> Tuple[np.ndarray, float]:
    x = hm.astype(np.float32)
    if np.isfinite(x).any():
        s = np.percentile(np.abs(x[np.isfinite(x)]), clip_abs_percentile)
    else:
        s = 1.0
    s = max(float(s), 1e-8)
    y = np.clip(x / s, -1.0, 1.0)
    return y, s


def overlay_heatmap(img: np.ndarray, heat: np.ndarray, alpha=0.5, cmap="jet", interp="bilinear", signed=False) -> np.ndarray:
    img_f = img.astype(np.float32) / 255.0
    H, W = img_f.shape[:2]
    heat = np.clip(heat.astype(np.float32), 0, 1) if not signed else np.clip(heat.astype(np.float32), -1, 1)

    if heat.shape[:2] != (H, W):
        heat_pil = Image.fromarray(((heat + 1) * 127.5).astype(np.uint8) if signed else (heat * 255).astype(np.uint8))
        heat = np.array(heat_pil.resize((W, H), resample=_pil_resample(interp))).astype(np.float32)
        if signed:
            heat = heat / 127.5 - 1.0
        else:
            heat = heat / 255.0

    cm = plt.get_cmap(cmap)
    if signed:
        colored = cm((heat + 1) / 2.0)[:, :, :3].astype(np.float32)
    else:
        colored = cm(heat)[:, :, :3].astype(np.float32)

    a = (alpha * np.abs(heat) if signed else alpha * heat).astype(np.float32)
    out = img_f * (1.0 - a[:, :, None]) + colored * a[:, :, None]
    return np.clip(out * 255, 0, 255).astype(np.uint8)


# ============================================================================
# Occlusion (signed) - prettier version (feather + interpolation)
# ============================================================================
def _gaussian_mask_2d(h: int, w: int, sigma_rel: float = 0.28) -> np.ndarray:
    """
    sigma_rel: relative to (h,w). 0.22~0.35 정도가 보통 예쁨.
    returns mask in [0,1], center ~1, edges ~0.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy = (yy - cy) / (h + 1e-8)
    xx = (xx - cx) / (w + 1e-8)
    g = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma_rel * sigma_rel))
    g = (g - g.min()) / (g.max() - g.min() + 1e-8)
    return g


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
    feather: bool = True,
    feather_sigma_rel: float = 0.28,
    upsample_order: int = 1,
    post_blur_sigma: float = 0.8,
) -> Tuple[np.ndarray, float]:
    model.eval()
    base_out = model(**batch, use_cache=False)
    base_logit = float(base_out["logits"][0, target_class].item())

    pv = batch["pixel_values"]
    if pv.dim() == 6:
        tile_sel = (0, representative_tile)
        _, _, _, _, H, W = pv.shape
    elif pv.dim() == 5:
        tile_sel = (representative_tile,)
        _, _, _, H, W = pv.shape
    elif pv.dim() == 4:
        tile_sel = None
        _, _, H, W = pv.shape
    else:
        raise ValueError(f"Unsupported pixel_values shape: {pv.shape}")

    patch_h = H // grid_h
    patch_w = W // grid_w
    block_h = patch_h * stride
    block_w = patch_w * stride
    step_h = max(1, grid_h // stride)
    step_w = max(1, grid_w // stride)

    coords = []
    for r in range(step_h):
        for c in range(step_w):
            rs = r * block_h
            re = min((r + 1) * block_h, H)
            cs = c * block_w
            ce = min((c + 1) * block_w, W)
            coords.append((r, c, rs, re, cs, ce))

    small = np.zeros((step_h, step_w), dtype=np.float32)

    # cache gaussian masks by (bh,bw)
    gcache: Dict[Tuple[int, int], torch.Tensor] = {}

    def apply_region_mask_(region: torch.Tensor, mv: float, gmask: Optional[torch.Tensor]):
        # region: [1,C,bh,bw]
        if gmask is None:
            region.fill_(mv)
        else:
            # region = region*(1-g) + mv*g
            region.mul_(1.0 - gmask).add_(gmask * mv)

    for i0 in tqdm(range(0, len(coords), micro_batch), desc=f"Occlusion (stride={stride})"):
        chunk = coords[i0:i0 + micro_batch]
        batch_masked = {}
        for k, v in batch.items():
            if not isinstance(v, torch.Tensor):
                batch_masked[k] = v
                continue

            if k == "pixel_values":
                vrep = v.repeat(len(chunk), *([1] * (v.dim() - 1)))

                for j, (_, _, rs, re, cs, ce) in enumerate(chunk):
                    bh = re - rs
                    bw = ce - cs

                    g_t = None
                    if feather:
                        key = (bh, bw)
                        if key not in gcache:
                            g = _gaussian_mask_2d(bh, bw, sigma_rel=feather_sigma_rel)
                            g_np = torch.from_numpy(g).to(vrep.device, dtype=vrep.dtype)  # [bh,bw]
                            gcache[key] = g_np
                        g = gcache[key]
                        # [1,C,bh,bw]
                        g_t = g.view(1, 1, bh, bw).repeat(1, vrep.shape[-3], 1, 1)

                    if pv.dim() == 6:
                        region = vrep[j:j+1, tile_sel[0], tile_sel[1], :, rs:re, cs:ce]
                        apply_region_mask_(region, mask_value, g_t)
                    elif pv.dim() == 5:
                        region = vrep[j:j+1, tile_sel[0], :, rs:re, cs:ce]
                        apply_region_mask_(region, mask_value, g_t)
                    else:
                        region = vrep[j:j+1, :, rs:re, cs:ce]
                        apply_region_mask_(region, mask_value, g_t)

                batch_masked[k] = vrep
            else:
                batch_masked[k] = v.repeat(len(chunk), *([1] * (v.dim() - 1)))

        out = model(**{kk: (vv.to(device) if isinstance(vv, torch.Tensor) else vv) for kk, vv in batch_masked.items()},
                    use_cache=False)
        masked_logits = out["logits"][:, target_class].float().cpu().numpy()

        for j, (r, c, *_rest) in enumerate(chunk):
            small[r, c] = base_logit - float(masked_logits[j])

        del batch_masked, out
        hard_cuda_gc(device)

    zoom = (grid_h / step_h, grid_w / step_w)
    occ = ndimage.zoom(small, zoom, order=upsample_order)
    if post_blur_sigma and post_blur_sigma > 0:
        occ = ndimage.gaussian_filter(occ, sigma=post_blur_sigma, mode="nearest")
    return occ.astype(np.float32), base_logit


# ============================================================================
# Save figure (5 panels)
# ============================================================================
def save_5panel_figure(
    out_path: str,
    original_img: Image.Image,
    panels: Sequence[Tuple[str, np.ndarray, bool]],
    cfg: RunConfig,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # keep your 2x3 layout, but hide the last slot
    fig, axes = plt.subplots(1, 5, figsize=(30, 15))
    axes = axes.reshape(-1)

    orig_np = np.array(original_img)
    for i in range(len(axes)):
        axes[i].axis("off")

    for idx, (title, heat, signed) in enumerate(panels):
        ax = axes[idx]
        if signed:
            heat_norm, _ = normalize_signed_map(heat)
            overlay = overlay_heatmap(orig_np, heat_norm, alpha=cfg.alpha, cmap="seismic", interp=cfg.interp, signed=True)
        else:
            heat_norm = normalize_map(heat, gamma=cfg.normalize_gamma, smooth_sigma=0.5)
            overlay = overlay_heatmap(orig_np, heat_norm, alpha=cfg.alpha, cmap=cfg.cmap, interp=cfg.interp, signed=False)

        ax.imshow(overlay)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--unique_id", required=True)
    parser.add_argument("--model_id", default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    parser.add_argument("--checkpoint_dir", default="../trained_models/dn+img")
    parser.add_argument("--out_dir", default="./heatmap_outputs")
    parser.add_argument("--meta_out_dir", default="./attention_outputs")

    parser.add_argument("--metadata_path", default="../dataset/metadata.json")
    parser.add_argument("--metadata_image_path", default="../dataset/test_summarization/full-test-indent-images.json")
    parser.add_argument("--test_data_path", default="../dataset/test_summarization/total_output.jsonl")
    parser.add_argument("--base_img_dir", default="../saved_images")
    parser.add_argument("--base_rr_dir", default="../physionet.org/files/mimic-cxr/2.1.0/files")

    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--cmap", default="jet")
    parser.add_argument("--summary_type", default="plain")
    parser.add_argument("--target_class", type=int, default=1)
    parser.add_argument("--interp", choices=["bilinear", "bicubic"], default="bilinear")
    parser.add_argument("--blur_radius", type=float, default=1.7)
    parser.add_argument("--patch_sigma", type=float, default=2.0)
    parser.add_argument("--normalize_gamma", type=float, default=1.2)

    parser.add_argument("--do_occlusion", action="store_true")
    parser.add_argument("--occ_stride", default=4, type=int)
    parser.add_argument("--occ_blocks", default=None, type=int)
    parser.add_argument("--occ_mask_value", default=0.0, type=float)
    parser.add_argument("--occ_micro_batch", default=8, type=int)

    parser.add_argument("--occ_feather", action="store_true")
    parser.add_argument("--occ_feather_sigma", type=float, default=0.30)
    parser.add_argument("--occ_upsample_order", type=int, default=1) 
    parser.add_argument("--occ_post_blur", type=float, default=0.9)

    parser.add_argument("--cross_layers", default=8, type=int)

    args = parser.parse_args()
    if args.occ_blocks is not None:
        args.occ_stride = int(args.occ_blocks)

    cfg = RunConfig(
        alpha=float(args.alpha),
        cmap=str(args.cmap),
        interp=str(args.interp),
        blur_radius=float(args.blur_radius),
        patch_sigma=float(args.patch_sigma),
        normalize_gamma=float(args.normalize_gamma),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(args.model_id)
    sample = load_sample_by_unique_id(args.unique_id, args, processor)

    from types import SimpleNamespace
    model_args = SimpleNamespace(
        model_name_or_path=args.model_id,
        checkpoint_dir=args.checkpoint_dir,
        inference=True,
        use_cxr_image=True,
        use_discharge_note=True,
        use_rad_report=False,
        use_generated_rad_report=False,
        use_pi=False,
        summarize=False,
        zeroshot=False,
        summary_type=args.summary_type,
        base_img_dir=args.base_img_dir,
        metadata_path=args.metadata_path,
        test_data_path=args.test_data_path,
        test_metadata_image_path=args.metadata_image_path,
    )

    model, _ = load_model(model_args, model_id=args.model_id, inference=True, attn_implementation="sdpa")
    model = model.to(device).eval()

    base_model = getattr(model, "base_model", model)
    vision_model = getattr(base_model, "vision_model", None)
    projector = getattr(base_model, "multi_modal_projector", None)

    if vision_model is None:
        raise RuntimeError("base_model.vision_model not found")
    if projector is None:
        raise RuntimeError("base_model.multi_modal_projector not found")

    # -----------------------
    # Build batch (no trunc)
    # -----------------------
    batch = build_batch(processor, sample, device)

    # -----------------------
    # Run inference + save meta json
    # -----------------------
    with torch.no_grad():
        out_inf = model(**batch, use_cache=False)
        logits = out_inf["logits"][0].float().cpu()
        prob = torch.softmax(logits, dim=-1).numpy()

    os.makedirs(args.meta_out_dir, exist_ok=True)
    meta_path = os.path.join(args.meta_out_dir, f"{args.unique_id}_summary.json")
    meta_obj = {
        "unique_id": sample["id"],
        "discharge_note": sample.get("discharge_note"),
        "radiology_report": sample.get("radiology_report"),
        "generated_radiology_report": sample.get("generated_radiology_report"),
        "image_path": sample.get("image_path"),
        "label": sample.get("label"),
        "prob_class_1": float(prob[1]) if prob.shape[0] > 1 else float(prob[0]),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_obj, f, ensure_ascii=False, indent=2)

    # -----------------------
    # Patch/grid info
    # -----------------------
    patch_size, grid_h, grid_w, ntiles, tile_h, tile_w = infer_grid_from_batch(batch, model)
    valid_tiles = get_valid_tile_indices(batch, ntiles)

    patch_mask = detect_padding_patches(batch["pixel_values"], ntiles, grid_h, grid_w, patch_size)

    mosaic_stat_mask_px = patchmask_to_mosaic_pixel_mask(
        patch_mask=patch_mask,
        tile_indices=valid_tiles,
        ntiles=ntiles,
        grid_h=grid_h,
        grid_w=grid_w,
        patch_size=patch_size,
        tile_h=tile_h,
        tile_w=tile_w,
    )

    def patchvals_to_heat_orig(patchvals_packed: List[np.ndarray]) -> np.ndarray:
        smoothed_patchvals = smooth_patchvals_packed(patchvals_packed, grid_h, grid_w, sigma=cfg.patch_sigma)

        heat_mosaic_px = tile_patchvals_to_mosaic_pixel_packed(
            patchvals_packed=smoothed_patchvals,
            tile_indices=valid_tiles,
            ntiles=ntiles,
            grid_h=grid_h,
            grid_w=grid_w,
            patch_size=patch_size,
            tile_h=tile_h,
            tile_w=tile_w,
        )

        heat_mosaic_px = normalize_map(
            heat_mosaic_px,
            stat_mask=mosaic_stat_mask_px,
            gamma=cfg.normalize_gamma,
            smooth_sigma=0.5,
        )

        heat_orig = mosaic_pixel_to_original(
            heat_mosaic_px,
            sample["image"],
            interp=cfg.interp,
            blur_radius_px=cfg.blur_radius,
            content_mask_px=mosaic_stat_mask_px,
            crop_to_content=True,
        )
        return heat_orig

    # =========================================================================
    # 1) Vision L0
    # 2) Projector norm
    # =========================================================================
    layer0_mod = get_vision_layer_module(vision_model, 0)

    cap_l0, h_l0 = install_capture(layer0_mod)
    cap_proj, h_proj = install_capture(projector)

    with torch.no_grad():
        _ = model(**batch, use_cache=False)

    h_l0.remove()
    h_proj.remove()

    if cap_l0.tensor is None:
        vision_l0_norm = np.zeros((grid_h, grid_w), dtype=np.float32)
    else:
        t = cap_l0.tensor
        token_norm = torch.linalg.norm(t.float(), dim=-1) if t.dim() >= 3 else torch.abs(t.float()).flatten()[None, :]
        l0_packed = scalar_to_tile_patchvals_packed(token_norm, grid_h, grid_w, ntiles, valid_tiles, patch_mask=patch_mask)
        vision_l0_norm = patchvals_to_heat_orig(l0_packed)

    if cap_proj.tensor is None:
        proj_norm = np.zeros((grid_h, grid_w), dtype=np.float32)
    else:
        t = cap_proj.tensor
        token_norm = torch.linalg.norm(t.float(), dim=-1) if t.dim() >= 3 else torch.abs(t.float()).flatten()[None, :]
        proj_packed = scalar_to_tile_patchvals_packed(token_norm, grid_h, grid_w, ntiles, valid_tiles, patch_mask=patch_mask)
        proj_norm = patchvals_to_heat_orig(proj_packed)

    hard_cuda_gc(device)

    # =========================================================================
    # 3) Self-Attn (Last tok -> Image tokens)  ✅ decoder-only는 self-attn
    # =========================================================================
    cross_attn = np.zeros((grid_h, grid_w), dtype=np.float32)
    
    vec = capture_llm_self_attn_lasttok_to_image_probs(
        base_model, batch, capture_last_n=args.cross_layers
    )
    if vec is not None and torch.is_tensor(vec):
        cross_packed = scalar_to_tile_patchvals_packed(
            vec[None, :], grid_h, grid_w, ntiles, valid_tiles, patch_mask=patch_mask
        )
        cross_attn = patchvals_to_heat_orig(cross_packed)
                

    hard_cuda_gc(device)

    # =========================================================================
    # 4) Self-Attn × Grad (Last tok -> Image tokens) ✅ 확률 상승 기여도(근사)
    #    ReLU(attn_probs * dlogit/dscores)  (Chefer-style)
    # =========================================================================
    llm_inp_x_grad = np.zeros((grid_h, grid_w), dtype=np.float32)
    
    vec = capture_llm_self_attn_lasttok_to_image_gradcam(
        model=model,
        base_model=base_model,
        batch=batch,
        target_class=args.target_class,
        capture_last_n=args.cross_layers,
    )

    if vec is not None and torch.is_tensor(vec):
        xg_packed = scalar_to_tile_patchvals_packed(
            vec[None, :], grid_h, grid_w, ntiles, valid_tiles, patch_mask=patch_mask
        )
        llm_inp_x_grad = patchvals_to_heat_orig(xg_packed)

    model.zero_grad(set_to_none=True)
    hard_cuda_gc(device)


    # =========================================================================
    # 5) Occlusion (signed) - prettier
    # =========================================================================
    if args.do_occlusion:
        occ_heat, _base_logit = compute_signed_occlusion_map(
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
            feather=bool(args.occ_feather),
            feather_sigma_rel=float(args.occ_feather_sigma),
            upsample_order=int(args.occ_upsample_order),
            post_blur_sigma=float(args.occ_post_blur),
        )
    else:
        occ_heat = np.zeros((grid_h, grid_w), dtype=np.float32)

    # upscale to original resolution for nicer viewing
    orig_w, orig_h = sample["image"].size
    occ_resized = ndimage.zoom(occ_heat, (orig_h / grid_h, orig_w / grid_w), order=3)

    # =========================================================================
    # Save panels (5)
    # =========================================================================
    panels = [
        ("Vision L0 Vector Norm", vision_l0_norm, False),
        ("Projector Vector Norm", proj_norm, False),
        ("Cross-Attn (Last Tok)", cross_attn, False),
        ("LLM Image Input × Grad", llm_inp_x_grad, False),
        ("Occlusion", occ_resized, True),
    ]

    out_path = os.path.join(args.out_dir, f"{args.unique_id}_5panel.png")
    save_5panel_figure(out_path, sample["image"], panels, cfg)

    print(f"Saved: {out_path}")
    print(f"Saved meta: {meta_path}")
    print(f"Settings: blur_radius={cfg.blur_radius}, patch_sigma={cfg.patch_sigma}, gamma={cfg.normalize_gamma}")

    del batch, model, processor
    hard_cuda_gc(device)


if __name__ == "__main__":
    main()
