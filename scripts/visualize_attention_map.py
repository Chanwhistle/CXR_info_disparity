#!/usr/bin/env python
"""
Visualize image regions that most influence the 0/1 classification for a TEST sample.

Uses Grad-CAM on either the vision transformer (ViT) or the multimodal projector.

Outputs (saved under --output_dir):
  - *_orig.png      : resized image used for visualization
  - *_saliency.png  : saliency heatmap
  - *_overlay.png   : overlay of heatmap on image
  - *_meta.json     : metadata (id/label/pred/prob/etc.)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Repo imports (scripts/ must be in sys.path)
# ---------------------------------------------------------------------------
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from dataloader import VLM_Dataset  # noqa: E402
from model import load_model  # noqa: E402
from utils import load_adapter, map_adapter_keys  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: pick last transformer block in vision encoder
# ---------------------------------------------------------------------------
def _pick_vision_target_layer(
    vision_model: nn.Module,
    use_local: bool = False,
    verbose: bool = True,
) -> nn.Module:
    """
    Best-effort: pick the last transformer block in the vision encoder.
    Works for many HF vision backbones including Mllama.

    Args:
        use_local: If True, prefer local/patch transformer (Mllama: transformer.layers).
                   If False, prefer global transformer (Mllama: global_transformer.layers).
    """
    # Priority order depends on use_local
    if use_local:
        paths = [
            ("transformer", "layers"),       # Mllama local (32 layers)
            ("global_transformer", "layers"),
            ("encoder", "layers"),
            ("layers",),
            ("blocks",),
        ]
    else:
        paths = [
            ("global_transformer", "layers"),  # Mllama global (8 layers)
            ("transformer", "layers"),
            ("encoder", "layers"),
            ("layers",),
            ("blocks",),
        ]

    for path in paths:
        m = vision_model
        ok = True
        for name in path:
            if hasattr(m, name):
                m = getattr(m, name)
            else:
                ok = False
                break
        if ok and isinstance(m, (nn.ModuleList, list)) and len(m) > 0:
            layer = m[-1]
            if verbose:
                print(f"[ViT] Selected layer via path {path}: {type(layer).__name__} (len={len(m)})")
            return layer

    # fallback: last child module with parameters
    for mod in reversed(list(vision_model.modules())):
        if isinstance(mod, nn.Module) and any(p.requires_grad for p in mod.parameters(recurse=False)):
            if verbose:
                print(f"[ViT] Fallback layer: {type(mod).__name__}")
            return mod

    raise ValueError("Could not find a suitable target layer in vision_model.")


# ---------------------------------------------------------------------------
# Grad-CAM for ViT-like token activations
# ---------------------------------------------------------------------------
class TokenGradCAM:
    """
    Grad-CAM for ViT-like token activations.
    Captures activations from a target layer and gradients w.r.t. target score,
    then computes CAM = ReLU(sum_c mean_grad_c * act_c) in token space.
    """

    def __init__(self, target_layer: nn.Module):
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_handle = None
        self.bwd_handle = None

    def _fwd_hook(self, module, inp, out):
        if isinstance(out, (tuple, list)):
            out = out[0]
        self.activations = out

    def _bwd_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __enter__(self):
        self.fwd_handle = self.target_layer.register_forward_hook(self._fwd_hook)
        self.bwd_handle = self.target_layer.register_full_backward_hook(self._bwd_hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.fwd_handle:
            self.fwd_handle.remove()
        if self.bwd_handle:
            self.bwd_handle.remove()

    def compute(self, score: torch.Tensor) -> torch.Tensor:
        """
        Backprop from score and return CAM grid (Hgrid, Wgrid).
        """
        if self.activations is None:
            raise RuntimeError("No activations captured.")
        self.target_layer.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        if self.gradients is None:
            raise RuntimeError("No gradients captured.")

        act = self.activations
        grad = self.gradients

        # Handle mllama projector output (B, I, T, S, D) -> flatten to (B, I*T*S, D)
        if act.dim() == 5:
            B, I, T, S, D = act.shape
            act = act.reshape(B, I * T * S, D)
            grad = grad.reshape(B, I * T * S, D)

        if act.dim() == 3:
            # (B, seq, dim) or (B, dim, seq)
            if act.shape[1] < act.shape[2]:
                tok, g = act, grad
            else:
                tok, g = act.transpose(1, 2), grad.transpose(1, 2)
        elif act.dim() == 4:
            # conv-like: (B, C, H, W)
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1).abs()  # abs instead of relu
            return cam[0].float()
        else:
            raise RuntimeError(f"Unexpected activation shape: {act.shape}")

        B, S, D = tok.shape

        # Remove CLS token if (S-1) is a perfect square
        s1 = S - 1
        r = int(math.sqrt(s1))
        if S >= 2 and r * r == s1:
            tok_, g_ = tok[:, 1:, :], g[:, 1:, :]
            S_use = s1
        else:
            tok_, g_, S_use = tok, g, S

        # Grad-CAM weights (use abs instead of ReLU for better ViT compatibility)
        weights = g_.mean(dim=1)  # (B, D)
        weighted_sum = (tok_ * weights[:, None, :]).sum(dim=-1)  # (B, S_use)
        cam_tok = weighted_sum.abs()  # abs works better than ReLU for deep layers

        # Reshape to grid
        grid = int(math.sqrt(S_use))
        if grid * grid != S_use:
            grid = int(math.ceil(math.sqrt(S_use)))
            pad = grid * grid - S_use
            cam_tok = torch.cat([cam_tok, cam_tok.new_zeros((B, pad))], dim=1)
        return cam_tok.view(B, grid, grid)[0].float()


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_jsonl(path: str, summary_type: str) -> list[dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            data.append({
                "id": s["id"],
                "label": s.get("label"),
                "text": s[summary_type],
                "summary_type": summary_type,
            })
    return data


def _resolve_real_image_path(dataset: VLM_Dataset, note_hash: str, base_img_dir: str) -> str:
    """Reconstruct the actual resized image path."""
    all_img_data = dataset.hash2meta[note_hash]["metadata_filtered"]
    best = dataset.Decision_tree.select_best_cxr(all_img_data)
    if best is None:
        raise ValueError(f"No CXR found for id={note_hash}")

    mapped_path = best[1]
    image_name = mapped_path.split("/")[-1]
    name, ext = image_name.split(".")
    real_name = image_name if "_512_resized" in name else f"{name}_512_resized.{ext}"

    if getattr(dataset, "split", None) is None:
        raise ValueError("Dataset split could not be determined.")

    real_path = os.path.join(base_img_dir, dataset.split, real_name)
    if not os.path.exists(real_path):
        raise FileNotFoundError(f"Image not found: {real_path}")
    return real_path


def _load_radiology_report_for_metadata(dataset: VLM_Dataset, note_hash: str, base_rr_dir: str) -> str | None:
    """Load radiology report text for metadata (regardless of --use_rad_report flag)."""
    try:
        all_img_data = dataset.hash2meta[note_hash]["metadata_filtered"]
        best = dataset.Decision_tree.select_best_cxr(all_img_data)
        if best is None:
            return None

        # Extract path parts for radiology report
        selected_img_path = best[1]
        path_parts = selected_img_path.split("/")[:3]
        if len(path_parts) != 3:
            return None

        rr_relative_path = "/".join(path_parts) + ".txt"
        rr_full_path = os.path.join(base_rr_dir, rr_relative_path)

        if os.path.exists(rr_full_path):
            with open(rr_full_path, "r", encoding="utf-8") as f:
                return f.read().replace("\n", " ").strip()
        return None
    except Exception:
        return None


def _normalize_map(
    x: np.ndarray,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
    gamma: float = 0.7,
    log_scale: bool = False,
) -> np.ndarray:
    """Robust visualization scaling with percentile clipping + gamma."""
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    x = np.maximum(x, 0.0)

    if log_scale:
        x = np.log1p(x)

    lo = float(np.percentile(x, clip_low)) if clip_low > 0.0 else float(x.min())
    hi = float(np.percentile(x, clip_high)) if clip_high < 100.0 else float(x.max())
    if hi <= lo + 1e-8:
        return np.zeros_like(x, dtype=np.float32)

    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo + 1e-8)

    if gamma is not None and abs(gamma - 1.0) > 1e-6:
        x = np.power(np.clip(x, 0.0, 1.0), gamma)

    return x.astype(np.float32)


def _create_overlay_array(
    image_rgb: Image.Image,
    heat: np.ndarray,
    alpha: float = 0.55,
    clip_low: float = 1.0,
    clip_high: float = 99.0,
    gamma: float = 0.7,
    log_scale: bool = False,
    cmap_name: str = "turbo",
    overlay_beta: float = 1.5,
) -> np.ndarray:
    """Create overlay image as numpy array [0,1] RGB."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = np.array(image_rgb).astype(np.float32) / 255.0
    H, W = img.shape[:2]

    heat_t = torch.tensor(heat)[None, None, ...]
    heat_t = F.interpolate(heat_t, size=(H, W), mode="bilinear", align_corners=False)
    heat_r = _normalize_map(heat_t[0, 0].cpu().numpy(), clip_low, clip_high, gamma, log_scale)

    cmap = plt.get_cmap(cmap_name)
    colored = cmap(heat_r)[..., :3]
    a = np.clip(alpha * (heat_r ** overlay_beta), 0.0, 1.0)
    overlay = np.clip(img * (1.0 - a[..., None]) + colored * a[..., None], 0, 1)

    return overlay


def _save_combined_horizontal(
    images: list,
    labels: list,
    out_path: str,
    dpi: int = 150,
) -> None:
    """Save multiple images as a single horizontal strip with labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def _load_checkpoint_into_model(model: nn.Module, checkpoint_path: str) -> None:
    """Load classifier + LoRA adapters (+ vision projector) from checkpoint."""
    cp = Path(checkpoint_path)
    if not cp.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # classifier
    model.classifier.load_state_dict(
        torch.load(cp / "classifier.bin", map_location="cpu", weights_only=True), strict=False
    )

    # language model LoRA
    lm_state = map_adapter_keys(
        torch.load(cp / "lm_adapter.bin", map_location="cpu", weights_only=False),
        "language_model_adapter",
    )
    lm_dict = model.base_model.language_model.state_dict()
    load_adapter(lm_dict, lm_state)
    model.base_model.language_model.load_state_dict(lm_dict, strict=False)

    # vision LoRA or encoder
    vm_path = cp / "vm_adapter.bin"
    ve_path = cp / "vision_encoder.bin"
    if vm_path.exists():
        vm_state = map_adapter_keys(
            torch.load(vm_path, map_location="cpu", weights_only=False),
            "vision_model_adapter",
        )
        vm_dict = model.base_model.vision_model.state_dict()
        load_adapter(vm_dict, vm_state)
        model.base_model.vision_model.load_state_dict(vm_dict, strict=False)
    elif ve_path.exists():
        model.base_model.vision_model.load_state_dict(
            torch.load(ve_path, map_location="cpu", weights_only=True), strict=False
        )

    # projector
    proj_path = cp / "multi_modal_projector.bin"
    if proj_path.exists():
        model.base_model.multi_modal_projector.load_state_dict(
            torch.load(proj_path, map_location="cpu", weights_only=True), strict=False
        )


@torch.no_grad()
def _predict_logits(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(**batch)["logits"]


def _gradcam_saliency(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    target_class: int,
    target_layer_name: str,
    verbose: bool = True,
) -> np.ndarray:
    """
    Grad-CAM saliency map.

    Args:
        target_layer_name: "projector", "vit" (global), or "vit_local"
    """
    model.zero_grad(set_to_none=True)
    torch.set_grad_enabled(True)

    if target_layer_name == "vit":
        target_layer = _pick_vision_target_layer(
            model.base_model.vision_model, use_local=False, verbose=verbose
        )
    elif target_layer_name == "vit_local":
        target_layer = _pick_vision_target_layer(
            model.base_model.vision_model, use_local=True, verbose=verbose
        )
    else:  # projector
        target_layer = model.base_model.multi_modal_projector
        if verbose:
            print(f"[Projector] Using: {type(target_layer).__name__}")

    with TokenGradCAM(target_layer) as cam:
        out = model(**batch)
        score = out["logits"][:, target_class].sum()
        cam_grid = cam.compute(score)

        # Debug: check activation/gradient stats
        if verbose and cam.activations is not None and cam.gradients is not None:
            act = cam.activations
            grad = cam.gradients
            print(f"[GradCAM] Activation shape: {tuple(act.shape)}, "
                  f"mean={float(act.abs().mean()):.6f}, max={float(act.abs().max()):.6f}")
            print(f"[GradCAM] Gradient shape: {tuple(grad.shape)}, "
                  f"mean={float(grad.abs().mean()):.6f}, max={float(grad.abs().max()):.6f}")
            print(f"[GradCAM] CAM grid shape: {tuple(cam_grid.shape)}, "
                  f"min={float(cam_grid.min()):.6f}, max={float(cam_grid.max()):.6f}")

    return cam_grid.detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Determine script directory for relative paths
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

    p = argparse.ArgumentParser(
        description="Visualize Grad-CAM saliency for classification on CXR images."
    )

    # Required
    p.add_argument("--checkpoint_path", type=str, required=True,
                   help="Directory containing classifier.bin/lm_adapter.bin/...")
    p.add_argument("--base_img_dir", type=str, required=True,
                   help="Path to saved_images folder (with train/dev/test subdirs)")

    # Core options with defaults
    p.add_argument("--target_layer", type=str, default="all",
                   choices=["all", "projector", "vit", "vit_local"],
                   help="Which layer(s) to visualize: 'all' for combined image, or single layer")
    p.add_argument("--index", type=int, default=1021,
                   help="Sample index (-1 for random)")
    p.add_argument("--target_class", type=int, default=1, choices=[0, 1],
                   help="0=alive, 1=death")
    p.add_argument("--output_dir", type=str, default="./attention_outputs")
    p.add_argument("--device", type=str, default="cuda")

    # Modality options (match your trained model)
    p.add_argument("--use_discharge_note", action="store_true", help="Use discharge note (for dn, dn+img, dn+rr models)")
    p.add_argument("--use_rad_report", action="store_true", help="Use radiology report (for rr, dn+rr models)")

    # Paths with repo-relative defaults
    p.add_argument("--metadata_path", type=str,
                   default=str(repo_root / "dataset" / "metadata.json"))
    p.add_argument("--test_data_path", type=str,
                   default=str(repo_root / "dataset" / "test_summarization" / "total_output.jsonl"))
    p.add_argument("--test_metadata_image_path", type=str,
                   default=str(repo_root / "dataset" / "test_summarization" / "full-test-indent-images.json"))
    p.add_argument("--base_rr_dir", type=str, 
                   default=str(repo_root / "physionet.org" / "files" / "mimic-cxr" / "2.1.0" / "files"))

    # Model/data
    p.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    p.add_argument("--summary_type", type=str, default="plain")
    p.add_argument("--seed", type=int, default=11)

    # Visualization
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.7)
    p.add_argument("--clip_low", type=float, default=10.0)
    p.add_argument("--clip_high", type=float, default=99.5)
    p.add_argument("--log_scale", action="store_true")
    p.add_argument("--cmap", type=str, default="turbo")
    p.add_argument("--overlay_beta", type=float, default=2.0)

    args = p.parse_args()
    _set_seed(args.seed)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Build repo-compatible args object
    @dataclass
    class RepoArgs:
        model_name_or_path: str
        checkpoint_dir: str
        summarize: bool = False
        zeroshot: bool = False
        use_cxr_image: bool = True
        use_rad_report: bool = False
        use_discharge_note: bool = False
        use_pi: bool = False
        summary_type: str = "plain"
        base_img_dir: str = "."
        base_rr_dir: str = "."
        metadata_path: str = ""
        test_data_path: str = ""
        test_metadata_image_path: str = ""

    rargs = RepoArgs(
        model_name_or_path=args.model_name_or_path,
        checkpoint_dir=str(Path(args.checkpoint_path).parent),
        use_cxr_image=True,  # Always True for visualization (need image for heatmap)
        use_rad_report=args.use_rad_report,
        use_discharge_note=args.use_discharge_note,
        summary_type=args.summary_type,
        base_img_dir=args.base_img_dir,
        base_rr_dir=args.base_rr_dir,
        metadata_path=args.metadata_path,
        test_data_path=args.test_data_path,
        test_metadata_image_path=args.test_metadata_image_path,
    )

    # Load model with proper checkpoint loading (inference=True)
    model, processor = load_model(rargs, model_id=rargs.model_name_or_path, inference=True)
    # Checkpoint is loaded inside load_model when inference=True

    try:
        model.to(device)
    except torch.cuda.OutOfMemoryError:
        print("[WARN] CUDA OOM, falling back to CPU.")
        device = torch.device("cpu")
        model.to(device)
    model.eval()

    # Load dataset
    test_data = _load_jsonl(args.test_data_path, args.summary_type)
    dataset = VLM_Dataset(
        rargs, test_data, args.test_metadata_image_path,
        use_cxr_image=True,  # Always need image for visualization
        use_rad_report=args.use_rad_report,
        use_discharge_note=args.use_discharge_note,
        shuffle=False,
    )

    idx = random.randrange(len(dataset)) if args.index < 0 else args.index
    ex = dataset[idx]
    sample_id = ex["id"]

    # Get image tensor from dataset (same as inference.py uses via collator)
    img_path = _resolve_real_image_path(dataset, sample_id, args.base_img_dir)
    pil_img = Image.open(img_path).convert("RGB")  # For visualization only

    # Prepare batch using dataset's pre-processed tensor image (same as custom_data_collator)
    text = processor.apply_chat_template(ex["chat_template"], tokenize=False)
    # Use tensor image from dataset, not PIL - this matches inference.py behavior
    tensor_img = ex["image"][0] if ex.get("image") else None
    if tensor_img is not None:
        batch = processor(text=[text], images=[[tensor_img]], return_tensors="pt", padding=True, truncation=False)
    else:
        batch = processor(text=[text], return_tensors="pt", padding=True, truncation=False)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    # Get predictions first
    with torch.no_grad():
        logits = _predict_logits(model, batch)[0].float().cpu().numpy()
        probs = F.softmax(torch.tensor(logits), dim=-1).numpy()
        pred = int(np.argmax(probs))

    # Resize image for visualization
    if "pixel_values" in batch:
        H, W = batch["pixel_values"].shape[-2], batch["pixel_values"].shape[-1]
        pil_viz = pil_img.resize((W, H), resample=Image.BILINEAR)
    else:
        pil_viz = pil_img

    # Original image as numpy array
    orig_arr = np.array(pil_viz).astype(np.float32) / 255.0

    # Determine which layers to compute
    if args.target_layer == "all":
        target_layers = ["projector", "vit", "vit_local"]
        layer_labels = ["Projector", "ViT (global)", "ViT (local)"]
    else:
        target_layers = [args.target_layer]
        layer_labels = [args.target_layer]

    # Compute Grad-CAM for target layers
    overlays = []
    for layer_name in target_layers:
        print(f"[INFO] Computing Grad-CAM for {layer_name}...")
        heat_raw = _gradcam_saliency(model, batch, args.target_class, layer_name, verbose=False)
        overlay = _create_overlay_array(
            pil_viz, heat_raw,
            alpha=args.alpha, clip_low=args.clip_low, clip_high=args.clip_high,
            gamma=args.gamma, log_scale=args.log_scale, cmap_name=args.cmap, overlay_beta=args.overlay_beta,
        )
        overlays.append(overlay)
        # Free GPU memory between computations
        del heat_raw
        model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Combine: Original + overlays (horizontal)
    images = [orig_arr] + overlays
    labels = ["Original"] + layer_labels

    suffix = "combined" if args.target_layer == "all" else args.target_layer
    out_path = os.path.join(args.output_dir, f"test_idx{idx}_id{sample_id}_t{args.target_class}_{suffix}.png")
    _save_combined_horizontal(images, labels, out_path)

    # Save metadata
    meta = {
        "idx": idx,
        "id": sample_id,
        "true_label": ex.get("label"),
        "target_class": args.target_class,
        "target_layer": args.target_layer,
        "pred": pred,
        "probs": probs.tolist(),
        "logits": logits.tolist(),
        "image_path": img_path,
        "use_discharge_note": args.use_discharge_note,
        "use_rad_report": args.use_rad_report,
        # Include text content for reference (always loaded for metadata)
        "discharge_note": ex.get("text"),  # From dataset (uses summary_type)
        "radiology_report": _load_radiology_report_for_metadata(dataset, sample_id, args.base_rr_dir),
    }
    meta_path = os.path.join(args.output_dir, f"test_idx{idx}_id{sample_id}_t{args.target_class}_{suffix}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()
