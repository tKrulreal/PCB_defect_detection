"""
GradCAM helpers tailored for the Streamlit UI.

Provides a self-contained API that:
  - Generates true Gradient-weighted Class Activation Maps (Grad-CAM) for YOLO defect crops.
  - Produces smooth, high-resolution JET heatmaps where defect regions are highlighted in vivid RED.
  - Generates high-contrast alpha-blended overlays preserving PCB trace context.
  - Returns clean RGB numpy arrays directly consumable by Streamlit `st.image`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import matplotlib
import numpy as np
import streamlit as st
import torch
from PIL import Image

# Use a non-interactive backend — must be set BEFORE importing pyplot.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gradcam_visualize import (  # noqa: E402
    GradCAM,
    _get_target_layer,
)


STAGE2_CLASSES = (
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
)


# ---------------------------------------------------------------------------
# Cached module-level singletons (one GradCAM per checkpoint / target layer)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_gradcam_for_bundle(_bundle_id: int, _bundle: dict) -> Optional[GradCAM]:
    """Return a long-lived :class:`GradCAM` instance for *_bundle*."""
    try:
        target_layer = _get_target_layer(_bundle["model"], _bundle["model_name"])
    except ValueError:
        return None
    return GradCAM(_bundle["model"], target_layer)


# ---------------------------------------------------------------------------
# Per-crop GradCAM generation
# ---------------------------------------------------------------------------

def _compute_crop_gradcam(crop_bgr: np.ndarray, bundle: dict, gradcam: GradCAM) -> dict:
    """Run GradCAM directly on a YOLO crop with high-fidelity colorization."""
    image_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]
    pil_image = Image.fromarray(image_rgb)

    device = next(bundle["model"].parameters()).device
    transform = bundle["transform"]
    tensor = transform(pil_image).unsqueeze(0).to(device)

    # 1. Forward pass → classification prediction
    bundle["model"].eval()
    with torch.no_grad():
        logits = bundle["model"](tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = probs.max(dim=1)
    
    pred_idx = int(pred_idx.item())
    confidence = float(confidence.item())
    pred_label = bundle["class_names"][pred_idx]

    # 2. Grad-CAM backward pass (requires gradients enabled on input)
    tensor_gc = transform(pil_image).unsqueeze(0).to(device)
    tensor_gc.requires_grad_(True)
    cam = gradcam(tensor_gc, class_idx=pred_idx)

    # 3. Smooth cubic interpolation to original crop resolution
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0.0, 1.0)

    # 4. Generate high-vibrancy RGB JET Heatmap (0.0 = Deep Blue, 1.0 = Intense Red)
    heatmap_rgb = (plt.cm.jet(cam_resized)[:, :, :3] * 255).astype(np.uint8)

    # 5. Create balanced Alpha-blended Overlay (55% original trace + 45% heatmap)
    overlay_rgb = cv2.addWeighted(image_rgb, 0.55, heatmap_rgb, 0.45, 0)

    return {
        "original": image_rgb,
        "cam": cam_resized,
        "heatmap": heatmap_rgb,
        "overlay": overlay_rgb,
        "pred_label": pred_label,
        "pred_idx": pred_idx,
        "confidence": confidence,
        "mode": "live",
    }


def render_gradcam_for_detection(
    crop_bgr: np.ndarray,
    image_stem: str,
    bundle: dict,
) -> dict:
    """Return a dict of high-resolution RGB images for one YOLO detection crop.

    Returns:
        dict: {
            "mode": "live"|"unsupported",
            "original": RGB numpy array,
            "cam": 2D float array in [0, 1],
            "heatmap": RGB JET heatmap numpy array (Red = Defect Peak),
            "overlay": RGB blended overlay numpy array,
            "pred_label": predicted class string,
            "confidence": float confidence score
        }
    """
    bundle_id = id(bundle)
    gradcam = get_gradcam_for_bundle(bundle_id, bundle)
    if gradcam is None:
        return {"mode": "unsupported", "original": None, "heatmap": None, "overlay": None}

    return _compute_crop_gradcam(crop_bgr, bundle, gradcam)


# ---------------------------------------------------------------------------
# Triple-panel matplotlib figure (used for export / reports)
# ---------------------------------------------------------------------------

def make_gradcam_triplet_figure(
    original_rgb: np.ndarray,
    cam: np.ndarray,
    overlay_rgb: np.ndarray,
    pred_label: str | None,
    confidence: float | None,
) -> plt.Figure:
    """Return a fresh matplotlib Figure with Original | Heatmap | Overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    h, w = original_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0.0, 1.0)

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Crop", fontweight="bold", fontsize=11)
    axes[0].axis("off")

    im = axes[1].imshow(cam_resized, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", fontweight="bold", fontsize=11)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay_rgb)
    axes[2].set_title("Blended Overlay", fontweight="bold", fontsize=11)
    axes[2].axis("off")

    if pred_label:
        title = f"AI Prediction: {pred_label}"
        if confidence is not None:
            title += f"  (Confidence: {confidence:.1%})"
        fig.suptitle(title, fontweight="bold", fontsize=12, y=0.98)

    fig.tight_layout()
    return fig
