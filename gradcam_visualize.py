"""
GradCAM visualization for Stage-2 CNN defect classifiers.

Generates class-activation heatmaps showing which image regions the CNN
attends to when making a prediction.

Usage:
    # Single image
    python gradcam_visualize.py --input demo_input/some_image.jpg \
                                --cnn runs/stage2/resnet18/best.pt

    # Directory of images (pick up to N)
    python gradcam_visualize.py --input pcb-defect-cls/test/mouse_bite/ \
                                --cnn runs/stage2/resnet18/best.pt \
                                --max-images 10

Output saved to runs/gradcam/
"""

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from stage2_cnn_utils import load_stage2_checkpoint

# ── Constants ────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("runs/gradcam")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Segoe UI", "DejaVu Sans"],
    "font.size": 11,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})


# ── GradCAM implementation ──────────────────────────────────────────────────

class GradCAM:
    """Gradient-weighted Class Activation Mapping.

    Hooks into the specified *target_layer* of the model, performs a forward
    pass, back-propagates the score of the predicted (or specified) class,
    and produces a spatial heat-map highlighting important regions.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    # ── Hook callbacks ───────────────────────────────────────────────────
    def _save_activation(self, _module, _input, output):
        self._activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    # ── Core ─────────────────────────────────────────────────────────────
    def __call__(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
    ) -> np.ndarray:
        """Return a (H, W) heat-map in [0, 1] for *input_tensor* (1, C, H, W)."""
        self.model.zero_grad()

        # Forward
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward for the target class
        target = output[0, class_idx]
        target.backward()

        # Global-average-pool the gradients → channel weights
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    # ── Cleanup ──────────────────────────────────────────────────────────
    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ── Target-layer selection ───────────────────────────────────────────────────

def _get_target_layer(model: torch.nn.Module, model_name: str) -> torch.nn.Module:
    """Return the last convolutional block to hook into."""
    if model_name in ("resnet18", "resnet50"):
        return model.layer4[-1]
    if model_name == "efficientnet_b2":
        return model.features[-1]
    raise ValueError(f"Unsupported model for GradCAM: {model_name}")


# ── Overlay helpers ──────────────────────────────────────────────────────────

def _apply_heatmap(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay *cam* heat-map on *image_rgb* using the 'jet' colourmap."""
    h, w = image_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    heatmap = plt.cm.jet(cam_resized)[:, :, :3]  # (H, W, 3) in [0,1]
    heatmap = (heatmap * 255).astype(np.uint8)

    overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap, alpha, 0)
    return overlay


def _save_triplet(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    overlay: np.ndarray,
    pred_label: str,
    confidence: float,
    save_path: Path,
) -> None:
    """Save a side-by-side figure: Original | Heatmap | Overlay."""
    h, w = image_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original", fontweight="bold")
    axes[0].axis("off")

    im = axes[1].imshow(cam_resized, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("GradCAM Heatmap", fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontweight="bold")
    axes[2].axis("off")

    fig.suptitle(
        f"Prediction: {pred_label}  (conf {confidence:.1%})",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path)
    plt.close(fig)


# ── Main logic ───────────────────────────────────────────────────────────────

def collect_image_paths(input_path: Path, max_images: int) -> list[Path]:
    """Resolve *input_path* to a list of image file paths."""
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        paths = sorted(
            p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        return paths[:max_images]
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def run_gradcam(
    image_paths: list[Path],
    checkpoint_path: Path,
    output_dir: Path,
) -> None:
    """Generate GradCAM visualizations for each image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {checkpoint_path}")
    bundle = load_stage2_checkpoint(checkpoint_path, device=device)
    model = bundle["model"]
    model_name = bundle["model_name"]
    class_names = bundle["class_names"]
    transform = bundle["transform"]

    target_layer = _get_target_layer(model, model_name)
    grad_cam = GradCAM(model, target_layer)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {output_dir}")
    print(f"Processing {len(image_paths)} image(s) …\n")

    for img_path in image_paths:
        # Load & preprocess
        pil_image = Image.open(img_path).convert("RGB")
        image_rgb = np.array(pil_image)
        input_tensor = transform(pil_image).unsqueeze(0).to(device)

        # Enable gradients for GradCAM (model is in eval mode)
        input_tensor.requires_grad_(False)

        # Forward pass → prediction
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = probs.max(dim=1)
        pred_idx = int(pred_idx.item())
        confidence = float(confidence.item())
        pred_label = class_names[pred_idx]

        # GradCAM (needs gradients, so re-enable)
        input_tensor_gc = transform(pil_image).unsqueeze(0).to(device)
        input_tensor_gc.requires_grad_(True)
        cam = grad_cam(input_tensor_gc, class_idx=pred_idx)

        # Overlay
        overlay = _apply_heatmap(image_rgb, cam)

        # Save
        stem = img_path.stem
        out_path = output_dir / f"{stem}_gradcam.png"
        _save_triplet(image_rgb, cam, overlay, pred_label, confidence, out_path)
        print(f"  [OK] {img_path.name} -> {pred_label} ({confidence:.1%})  ->  {out_path.name}")

    grad_cam.remove_hooks()
    print(f"\nDone – {len(image_paths)} visualisation(s) saved.")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GradCAM heatmaps for Stage-2 CNN classifiers.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a single image or a directory of images.",
    )
    parser.add_argument(
        "--cnn",
        type=str,
        required=True,
        help="Path to Stage-2 CNN checkpoint (best.pt).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=20,
        help="Maximum number of images to process when input is a directory (default: 20).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output directory (default: runs/gradcam/).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    checkpoint_path = Path(args.cnn)
    output_dir = Path(args.output) if args.output else OUTPUT_DIR

    image_paths = collect_image_paths(input_path, args.max_images)
    if not image_paths:
        print(f"No images found at {input_path}")
        return

    run_gradcam(image_paths, checkpoint_path, output_dir)


if __name__ == "__main__":
    main()
