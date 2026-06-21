"""
Visualize Stage-2 CNN training results.

Reads confusion-matrix and history CSVs produced by training scripts and
generates publication-quality plots saved alongside the original files.

Usage:
    python visualize_results.py --model resnet18
    python visualize_results.py --model all
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ── Constants ────────────────────────────────────────────────────────────────
STAGE2_DIR = Path("runs/stage2")
KNOWN_MODELS = ["resnet18", "resnet50", "efficientnet_b2"]
DEFAULT_CLASSES = sorted([
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
])

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Segoe UI", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

PALETTE = {
    "train_loss": "#e74c3c",
    "val_loss": "#3498db",
    "train_acc": "#e67e22",
    "val_acc": "#2ecc71",
    "val_f1": "#9b59b6",
    "lr": "#1abc9c",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pretty_model_name(model: str) -> str:
    """Return a human-friendly model name for plot titles."""
    mapping = {
        "resnet18": "ResNet-18",
        "resnet50": "ResNet-50",
        "efficientnet_b2": "EfficientNet-B2",
    }
    return mapping.get(model, model)


def _load_class_names(model_dir: Path) -> list[str]:
    """Try to load class names from best.pt checkpoint; fall back to defaults."""
    checkpoint_path = model_dir / "best.pt"
    if checkpoint_path.exists():
        try:
            import torch
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if "idx_to_class" in checkpoint:
                idx_to_class = checkpoint["idx_to_class"]
                return [idx_to_class[i] for i in sorted(idx_to_class.keys())]
            if "class_to_idx" in checkpoint:
                class_to_idx = checkpoint["class_to_idx"]
                return [n for n, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
        except Exception:
            pass
    return DEFAULT_CLASSES


# ── Confusion-matrix plot ────────────────────────────────────────────────────

def plot_confusion_matrix(model_dir: Path, model: str) -> None:
    """Load the raw confusion-matrix CSV and render a seaborn heatmap."""
    cm_path = model_dir / "test_confusion_matrix.csv"
    if not cm_path.exists():
        print(f"  [SKIP] Confusion matrix not found: {cm_path}")
        return

    cm = np.loadtxt(cm_path, delimiter=",", dtype=int)
    class_names = _load_class_names(model_dir)

    # Normalise for percentage display (keep raw counts as annotations)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, cm / row_sums * 100, 0.0)

    # Build annotation strings: "count\n(pct%)"
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        cm_pct,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Recall (%)"},
        ax=ax,
        vmin=0,
        vmax=100,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix — {_pretty_model_name(model)}")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()

    out = model_dir / "confusion_matrix.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] Saved confusion matrix -> {out}")


# ── Training-curve plots ────────────────────────────────────────────────────

def plot_training_curves(model_dir: Path, model: str) -> None:
    """Plot loss, accuracy, F1 and LR curves from history.csv."""
    history_path = model_dir / "history.csv"
    if not history_path.exists():
        print(f"  [SKIP] History not found: {history_path}")
        return

    df = pd.read_csv(history_path)
    epochs = df["epoch"]
    title_prefix = _pretty_model_name(model)

    # ── 2×2 grid ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Training Curves — {title_prefix}", fontsize=16, fontweight="bold", y=0.99)

    # 1) Loss
    ax = axes[0, 0]
    ax.plot(epochs, df["train_loss"], color=PALETTE["train_loss"], linewidth=1.8, label="Train loss")
    ax.plot(epochs, df["val_loss"], color=PALETTE["val_loss"], linewidth=1.8, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # 2) Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, df["train_acc"] * 100, color=PALETTE["train_acc"], linewidth=1.8, label="Train acc")
    ax.plot(epochs, df["val_acc"] * 100, color=PALETTE["val_acc"], linewidth=1.8, label="Val acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy")
    ax.legend(framealpha=0.9, loc="lower right")
    ax.grid(True, alpha=0.3)

    # 3) F1 Score
    ax = axes[1, 0]
    ax.plot(epochs, df["val_macro_f1"] * 100, color=PALETTE["val_f1"], linewidth=1.8, label="Val macro-F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1 (%)")
    ax.set_title("Validation Macro-F1")
    ax.legend(framealpha=0.9, loc="lower right")
    ax.grid(True, alpha=0.3)

    # 4) Learning rate
    ax = axes[1, 1]
    ax.plot(epochs, df["lr"], color=PALETTE["lr"], linewidth=1.8, label="Learning rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.set_title("Learning Rate Schedule")
    ax.legend(framealpha=0.9)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(-3, -3))
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = model_dir / "training_curves.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] Saved training curves -> {out}")


# ── Per-model driver ─────────────────────────────────────────────────────────

def visualize_model(model: str) -> None:
    model_dir = STAGE2_DIR / model
    if not model_dir.exists():
        print(f"[WARN] Directory not found, skipping: {model_dir}")
        return
    print(f"\n-- {_pretty_model_name(model)} --")
    plot_confusion_matrix(model_dir, model)
    plot_training_curves(model_dir, model)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-quality plots from Stage-2 training results.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model name (resnet18 | resnet50 | efficientnet_b2) or 'all'.",
    )
    args = parser.parse_args()

    if args.model == "all":
        models = KNOWN_MODELS
    else:
        models = [args.model]

    for model in models:
        visualize_model(model)

    print("\nDone.")


if __name__ == "__main__":
    main()
