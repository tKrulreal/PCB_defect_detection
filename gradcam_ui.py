"""
GradCAM helpers tailored for the Streamlit UI.

Provides a small, self-contained API that:
  - generates GradCAM heatmaps for individual YOLO-detected crops
  - returns RGB numpy arrays suitable for ``st.image``
  - caches pre-computed visualizations from ``runs/gradcam/`` when matching
    filenames are available

Re-uses primitives from :mod:`gradcam_visualize` so behavior stays identical
to the offline CLI workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import matplotlib
import numpy as np
import streamlit as st

# Use a non-interactive backend — must be set BEFORE importing pyplot.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gradcam_visualize import (  # noqa: E402
    GradCAM,
    _apply_heatmap,
    _get_target_layer,
)


GRADCAM_CACHE_DIR = Path("runs/gradcam")
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
def get_gradcam_for_bundle(_bundle_id: int, bundle: dict) -> Optional[GradCAM]:
    """Return a long-lived :class:`GradCAM` instance for *bundle*.

    The first arg is a coarse ``id(bundle)`` to defeat Streamlit's cache
    equality check (the bundle contains non-hashable tensors).  The GradCAM
    hooks attach to ``bundle["model"]``, so a fresh helper is required when
    the user switches CNN checkpoints.
    """
    try:
        target_layer = _get_target_layer(bundle["model"], bundle["model_name"])
    except ValueError:
        return None
    return GradCAM(bundle["model"], target_layer)


# ---------------------------------------------------------------------------
# Per-crop GradCAM generation
# ---------------------------------------------------------------------------

def _find_cached_heatmap(stem: str) -> Optional[Path]:
    """Look for an already-generated ``runs/gradcam/<stem>_gradcam.png``."""
    candidate = GRADCAM_CACHE_DIR / f"{stem}_gradcam.png"
    return candidate if candidate.exists() else None


def _compute_crop_gradcam(crop_bgr: np.ndarray, bundle: dict, gradcam: GradCAM):
    """Run GradCAM directly on a YOLO crop (no matplotlib triplet)."""
    from PIL import Image

    image_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    transform = bundle["transform"]
    tensor = transform(pil_image).unsqueeze(0).to(
        next(bundle["model"].parameters()).device
    )

    # Forward → predict
    bundle["model"].eval()
    with torch.no_grad():  # noqa: F821 -- guarded by import below
        logits = bundle["model"](tensor)
        probs = torch.softmax(logits, dim=1)  # noqa: F821
        confidence, pred_idx = probs.max(dim=1)
    pred_idx = int(pred_idx.item())
    confidence = float(confidence.item())
    pred_label = bundle["class_names"][pred_idx]

    # GradCAM (needs gradients)
    tensor_gc = transform(pil_image).unsqueeze(0).to(
        next(bundle["model"].parameters()).device
    )
    tensor_gc.requires_grad_(True)
    cam = gradcam(tensor_gc, class_idx=pred_idx)
    overlay_rgb = _apply_heatmap(image_rgb, cam)

    return {
        "original": image_rgb,
        "cam": cam,
        "overlay": overlay_rgb,
        "pred_label": pred_label,
        "pred_idx": pred_idx,
        "confidence": confidence,
    }


def render_gradcam_for_detection(
    crop_bgr: np.ndarray,
    image_stem: str,
    bundle: dict,
) -> dict:
    """Return a dict of RGB images for one YOLO detection.

    Tries (in order):
      1. Reuse cached figure at ``runs/gradcam/<stem>_gradcam.png`` if its
         filename happens to match (best-effort, log-only fallback).
      2. Run live GradCAM on the crop using ``bundle``.

    Returns ``{"mode": "live"|"cached", "original", "overlay", "cam",
    "pred_label", "confidence"}``.
    """
    cached = _find_cached_heatmap(image_stem)
    if cached is not None:
        # Read the cached triplet figure and slice out the overlay panel.
        # Falls back silently if reading fails — live compute continues.
        try:
            fig = plt.imread(str(cached))
            if fig.ndim == 3 and fig.shape[1] >= 3:
                # Middle panel ≈ heatmap; right panel ≈ overlay (heuristic).
                overlay_rgb = (fig[:, 2 * fig.shape[1] // 3 :, :] * 255).astype(
                    np.uint8
                )
                original_rgb = None  # live run will populate.
                return {
                    "mode": "cached",
                    "original": original_rgb,
                    "overlay": overlay_rgb,
                    "cam": None,
                    "pred_label": None,
                    "confidence": None,
                }
        except Exception:  # pragma: no cover -- defensive
            pass

    import torch  # local import keeps cold-import cheap

    bundle_id = id(bundle)
    gradcam = get_gradcam_for_bundle(bundle_id, bundle)
    if gradcam is None:
        return {"mode": "unsupported", "original": None, "overlay": None}

    out = _compute_crop_gradcam(crop_bgr, bundle, gradcam)
    out["mode"] = "live"
    return out


# ---------------------------------------------------------------------------
# Triple-panel matplotlib figure (used inside expander for low-token view)
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
    cam_resized = cv2.resize(cam, (w, h))

    axes[0].imshow(original_rgb)
    axes[0].set_title("Crop", fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(cam_resized, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("GradCAM", fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(overlay_rgb)
    axes[2].set_title("Overlay", fontweight="bold")
    axes[2].axis("off")

    if pred_label:
        title = f"Prediction: {pred_label}"
        if confidence is not None:
            title += f"  ({confidence:.1%})"
        fig.suptitle(title, fontweight="bold", y=1.02)

    fig.tight_layout()
    return fig
