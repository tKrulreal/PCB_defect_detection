"""Error analysis for the YOLO + CNN pipeline.

Reads pipeline_errors.csv and ground-truth labels to produce detailed
error breakdowns by defect class, error type, and bbox size. Generates
publication-quality charts and example-image grids for the thesis report.

Usage:
    python error_analysis.py
    python error_analysis.py --eval-dir runs/system_eval/resnet18_test
    python error_analysis.py --eval-dir runs/system_eval/efficientnet_b2_test
    python error_analysis.py --all
"""

import argparse
import json
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import load_dataset_config, resolve_split_dirs, yolo_line_to_xyxy

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Segoe UI", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})

DEFECT_COLORS = {
    "missing_hole": "#ef4444",
    "mouse_bite": "#f97316",
    "open_circuit": "#eab308",
    "short": "#22c55e",
    "spur": "#3b82f6",
    "spurious_copper": "#8b5cf6",
}

ERROR_COLORS = {
    "missed_detection": "#ef4444",
    "false_positive_detection": "#f97316",
    "misclassification": "#8b5cf6",
}

DATA_YAML = "pcb-defect-dataset/data.yaml"
EVAL_DIRS = {
    "resnet18": "runs/system_eval/resnet18_test",
    "resnet50": "runs/system_eval/resnet50_test",
    "efficientnet_b2": "runs/system_eval/efficientnet_b2_test",
}


# ── GT statistics ────────────────────────────────────────────────────────────

def compute_gt_stats(data_yaml, split="test"):
    """Count GT boxes per class and compute bbox area distribution."""
    dataset_root, data_cfg, class_names = load_dataset_config(data_yaml)
    images_dir, labels_dir = resolve_split_dirs(dataset_root, data_cfg[split])

    class_counts = Counter()
    bbox_areas = defaultdict(list)  # class_name -> list of relative areas

    for label_file in sorted(labels_dir.glob("*.txt")):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            w_rel = float(parts[3])
            h_rel = float(parts[4])
            area_rel = w_rel * h_rel

            cls_name = class_names[cls_id]
            class_counts[cls_name] += 1
            bbox_areas[cls_name].append(area_rel)

    return class_names, class_counts, bbox_areas


# ── Error breakdown ──────────────────────────────────────────────────────────

def load_errors(eval_dir):
    """Load pipeline_errors.csv into a DataFrame."""
    csv_path = Path(eval_dir) / "pipeline_errors.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_summary(eval_dir):
    """Load system_summary.json."""
    json_path = Path(eval_dir) / "system_summary.json"
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def error_by_class_and_type(df, class_names):
    """Build a pivot table: class x error_type."""
    # For missed_detection, the class is gt_label
    # For false_positive, the class is pred_label
    # For misclassification, the class is gt_label
    rows = []
    for _, row in df.iterrows():
        error_type = row["error_type"]
        if error_type == "false_positive_detection":
            cls = row["pred_label"] if pd.notna(row["pred_label"]) else "unknown"
        else:
            cls = row["gt_label"] if pd.notna(row["gt_label"]) else "unknown"
        rows.append({"class": cls, "error_type": error_type})

    detail_df = pd.DataFrame(rows)
    pivot = detail_df.pivot_table(
        index="class", columns="error_type", aggfunc="size", fill_value=0,
    )

    # Ensure all classes and error types present
    for cls in class_names:
        if cls not in pivot.index:
            pivot.loc[cls] = 0
    for et in ["missed_detection", "false_positive_detection", "misclassification"]:
        if et not in pivot.columns:
            pivot[et] = 0

    pivot = pivot.reindex(sorted(class_names))
    pivot = pivot[["missed_detection", "false_positive_detection", "misclassification"]]
    pivot["total_errors"] = pivot.sum(axis=1)
    return pivot


def misclassification_matrix(df, class_names):
    """Build confusion pairs for misclassifications: gt_label -> pred_label."""
    misclass = df[df["error_type"] == "misclassification"]
    pairs = Counter()
    for _, row in misclass.iterrows():
        gt = row["gt_label"]
        pred = row["pred_label"]
        if pd.notna(gt) and pd.notna(pred):
            pairs[(gt, pred)] += 1
    return pairs


# ── Visualization ────────────────────────────────────────────────────────────

def plot_error_by_class(pivot, model_name, output_dir):
    """Stacked bar chart: errors by class and type."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    classes = list(pivot.index)
    x = np.arange(len(classes))
    width = 0.55

    missed = pivot["missed_detection"].values
    fp = pivot["false_positive_detection"].values
    misclass = pivot["misclassification"].values

    ax.bar(x, missed, width, label="Missed Detection", color=ERROR_COLORS["missed_detection"], alpha=0.85)
    ax.bar(x, fp, width, bottom=missed, label="False Positive", color=ERROR_COLORS["false_positive_detection"], alpha=0.85)
    ax.bar(x, misclass, width, bottom=missed + fp, label="Misclassification", color=ERROR_COLORS["misclassification"], alpha=0.85)

    # Value labels
    for i, total in enumerate(pivot["total_errors"].values):
        if total > 0:
            ax.text(i, total + 2, str(total), ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Number of Errors")
    ax.set_title(f"Pipeline Errors by Defect Class ({model_name})")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = output_dir / "error_by_class.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


def plot_error_type_pie(df, model_name, output_dir):
    """Pie chart of error types."""
    counts = df["error_type"].value_counts()

    labels_map = {
        "missed_detection": "Missed Detection",
        "false_positive_detection": "False Positive",
        "misclassification": "Misclassification",
    }
    labels = [labels_map.get(k, k) for k in counts.index]
    colors = [ERROR_COLORS.get(k, "#94a3b8") for k in counts.index]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct=lambda pct: f"{pct:.1f}%\n({int(round(pct * sum(counts.values) / 100))})",
        startangle=90,
        textprops={"fontsize": 11},
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for autotext in autotexts:
        autotext.set_fontweight("bold")

    ax.set_title(f"Error Type Distribution ({model_name})")
    fig.tight_layout()

    out = output_dir / "error_type_pie.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


def plot_gt_distribution(class_names, class_counts, output_dir):
    """Bar chart of GT box counts per class."""
    classes = sorted(class_names)
    counts = [class_counts[c] for c in classes]
    colors = [DEFECT_COLORS.get(c, "#94a3b8") for c in classes]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(classes, counts, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.set_ylabel("Number of Ground Truth Boxes")
    ax.set_title("Ground Truth Distribution (test split)")
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = output_dir / "gt_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


def plot_miss_rate_by_class(pivot, class_counts, class_names, model_name, output_dir):
    """Bar chart: miss rate (%) per class."""
    classes = sorted(class_names)
    miss_rates = []
    for cls in classes:
        missed = pivot.loc[cls, "missed_detection"] if cls in pivot.index else 0
        total = class_counts[cls]
        rate = (missed / total * 100) if total > 0 else 0
        miss_rates.append(rate)

    colors = [DEFECT_COLORS.get(c, "#94a3b8") for c in classes]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(classes, miss_rates, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)

    for bar, rate in zip(bars, miss_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{rate:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.set_ylabel("Miss Rate (%)")
    ax.set_title(f"YOLO Stage 1 Miss Rate by Defect Class ({model_name})")
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylim(0, max(miss_rates) * 1.2 if miss_rates else 100)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = output_dir / "miss_rate_by_class.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


def plot_misclassification_heatmap(pairs, class_names, model_name, output_dir):
    """Small heatmap of misclassification pairs (gt -> pred)."""
    if not pairs:
        return

    classes = sorted(class_names)
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for (gt, pred), count in pairs.items():
        if gt in classes and pred in classes:
            i = classes.index(gt)
            j = classes.index(pred)
            matrix[i, j] = count

    # Only show rows/cols that have errors
    has_data = (matrix.sum(axis=1) > 0) | (matrix.sum(axis=0) > 0)
    if not has_data.any():
        return

    sub_classes = [c for c, h in zip(classes, has_data) if h]
    sub_idx = [classes.index(c) for c in sub_classes]
    sub_matrix = matrix[np.ix_(sub_idx, sub_idx)]

    fig, ax = plt.subplots(figsize=(7, 5.5))

    import seaborn as sns
    sns.heatmap(
        sub_matrix, annot=True, fmt="d", cmap="Reds",
        xticklabels=sub_classes, yticklabels=sub_classes,
        linewidths=0.6, linecolor="white",
        cbar_kws={"label": "Count"}, ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Misclassification Pairs ({model_name})")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()

    out = output_dir / "misclassification_heatmap.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


def save_example_images(df, data_yaml, split, output_dir, max_per_type=6):
    """Save grid images showing examples of each error type."""
    dataset_root, data_cfg, class_names = load_dataset_config(data_yaml)
    images_dir, _ = resolve_split_dirs(dataset_root, data_cfg[split])

    for error_type in ["missed_detection", "false_positive_detection", "misclassification"]:
        subset = df[df["error_type"] == error_type]
        if subset.empty:
            continue

        # Get unique images
        unique_images = subset["image"].unique()[:max_per_type]
        imgs = []
        titles = []

        for img_name in unique_images:
            img_path = images_dir / "images" / img_name
            if not img_path.exists():
                img_path = images_dir / img_name
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize for display
            h, w = img.shape[:2]
            scale = 300 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            imgs.append(img)

            err_row = subset[subset["image"] == img_name].iloc[0]
            if error_type == "missed_detection":
                titles.append(f"GT: {err_row['gt_label']}")
            elif error_type == "false_positive_detection":
                titles.append(f"FP: {err_row['pred_label']}\nconf={err_row['stage1_confidence']:.2f}")
            else:
                titles.append(f"{err_row['gt_label']}\n-> {err_row['pred_label']}")

        if not imgs:
            continue

        ncols = min(len(imgs), 3)
        nrows = (len(imgs) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)

        type_name = error_type.replace("_", " ").title()
        fig.suptitle(f"Example: {type_name}", fontsize=14, fontweight="bold", y=1.02)

        for idx, (img, title) in enumerate(zip(imgs, titles)):
            r, c = divmod(idx, ncols)
            axes[r, c].imshow(img)
            axes[r, c].set_title(title, fontsize=10)
            axes[r, c].axis("off")

        # Hide empty axes
        for idx in range(len(imgs), nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].axis("off")

        fig.tight_layout()
        out = output_dir / f"examples_{error_type}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  [OK] {out}")


# ── Report generation ────────────────────────────────────────────────────────

def generate_report(pivot, pairs, class_counts, summary, model_name, output_dir):
    """Generate a markdown error analysis report."""
    total_gt = summary["system"]["total_gt_boxes"] if summary else "N/A"
    total_pred = summary["system"]["total_predictions"] if summary else "N/A"
    system_acc = summary["system"]["overall_accuracy"] if summary else 0

    classes = sorted(pivot.index)

    # Error by class table
    class_table_rows = []
    for cls in classes:
        gt = class_counts[cls]
        missed = pivot.loc[cls, "missed_detection"]
        fp = pivot.loc[cls, "false_positive_detection"]
        misclass = pivot.loc[cls, "misclassification"]
        miss_rate = (missed / gt * 100) if gt > 0 else 0
        total_err = pivot.loc[cls, "total_errors"]
        class_table_rows.append(
            f"| {cls} | {gt} | {missed} | {miss_rate:.1f}% | {fp} | {misclass} | {total_err} |"
        )

    # Misclassification pairs
    pair_lines = []
    for (gt, pred), count in sorted(pairs.items(), key=lambda x: -x[1]):
        pair_lines.append(f"| {gt} | {pred} | {count} |")

    report = textwrap.dedent(f"""\
    # Error Analysis Report — {model_name}

    ## Overview

    | Metric | Value |
    | --- | --- |
    | Total GT boxes | {total_gt} |
    | Total predictions | {total_pred} |
    | System accuracy | {system_acc:.4f} |
    | Missed detections | {pivot['missed_detection'].sum()} |
    | False positives | {pivot['false_positive_detection'].sum()} |
    | Misclassifications | {pivot['misclassification'].sum()} |

    ## Errors by Defect Class

    | Class | GT Count | Missed | Miss Rate | False Positive | Misclassified | Total Errors |
    | --- | --- | --- | --- | --- | --- | --- |
    {chr(10).join(class_table_rows)}

    ## Misclassification Pairs (GT -> Predicted)

    | True Label | Predicted Label | Count |
    | --- | --- | --- |
    {chr(10).join(pair_lines) if pair_lines else "| (none) | — | — |"}

    ## Key Findings

    1. **Bottleneck**: Stage 1 (YOLO) missed {pivot['missed_detection'].sum()} out of {total_gt} GT boxes ({pivot['missed_detection'].sum() / total_gt * 100:.1f}%), which is the primary factor limiting system accuracy.
    2. **Stage 2 performs well**: Only {pivot['misclassification'].sum()} misclassifications out of {summary['classification']['num_detected_boxes'] if summary else 'N/A'} detected boxes ({summary['classification']['accuracy_on_detected_boxes'] * 100:.2f}% accuracy).
    3. **Most confused pair**: {f'{sorted(pairs.items(), key=lambda x: -x[1])[0][0][0]} -> {sorted(pairs.items(), key=lambda x: -x[1])[0][0][1]} ({sorted(pairs.items(), key=lambda x: -x[1])[0][1]} times)' if pairs else 'N/A'}

    ## Generated Visualizations

    - `error_by_class.png` — Stacked bar: errors by class and type
    - `error_type_pie.png` — Pie chart of error distribution
    - `miss_rate_by_class.png` — YOLO miss rate per class
    - `misclassification_heatmap.png` — GT vs Predicted confusion
    - `gt_distribution.png` — GT class distribution
    - `examples_*.png` — Sample images for each error type
    """)

    out = output_dir / "error_analysis_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"  [OK] {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def analyze_one(eval_dir, model_name, class_names, class_counts, bbox_areas):
    """Run full error analysis for one evaluation directory."""
    eval_dir = Path(eval_dir)
    if not eval_dir.exists():
        print(f"[SKIP] {eval_dir} not found")
        return

    output_dir = eval_dir / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n== {model_name} ==")
    print(f"   Source: {eval_dir}")
    print(f"   Output: {output_dir}")

    df = load_errors(eval_dir)
    summary = load_summary(eval_dir)
    pivot = error_by_class_and_type(df, class_names)
    pairs = misclassification_matrix(df, class_names)

    # Generate all charts
    plot_error_by_class(pivot, model_name, output_dir)
    plot_error_type_pie(df, model_name, output_dir)
    plot_gt_distribution(class_names, class_counts, output_dir)
    plot_miss_rate_by_class(pivot, class_counts, class_names, model_name, output_dir)
    plot_misclassification_heatmap(pairs, class_names, model_name, output_dir)
    save_example_images(df, DATA_YAML, "test", output_dir)
    generate_report(pivot, pairs, class_counts, summary, model_name, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Error analysis for PCB pipeline")
    parser.add_argument("--eval-dir", type=str, default=None, help="Specific eval directory")
    parser.add_argument("--all", action="store_true", help="Analyze all available models")
    args = parser.parse_args()

    # Compute GT stats once
    class_names, class_counts, bbox_areas = compute_gt_stats(DATA_YAML, "test")
    print(f"GT distribution: {dict(class_counts)}")

    if args.all:
        for model_name, eval_dir in EVAL_DIRS.items():
            analyze_one(eval_dir, model_name, class_names, class_counts, bbox_areas)
    elif args.eval_dir:
        model_name = Path(args.eval_dir).name
        analyze_one(args.eval_dir, model_name, class_names, class_counts, bbox_areas)
    else:
        # Default: analyze resnet18
        default_dir = EVAL_DIRS.get("resnet18", "runs/system_eval/resnet18_test")
        analyze_one(default_dir, "ResNet18", class_names, class_counts, bbox_areas)

    print("\nDone.")


if __name__ == "__main__":
    main()
