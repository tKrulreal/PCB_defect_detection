#  """Shared utilities for the PCB defect detection project.
#
# This module consolidates common functions previously duplicated across
# stage2_crop_dataset.py, evaluate_stage12_system.py, and training scripts.
# """

from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def seed_everything(seed):
    """Set random seeds for reproducibility across random, numpy, and torch."""
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Dataset / YAML helpers
# ---------------------------------------------------------------------------


def load_dataset_config(data_yaml_path):
    """Parse a YOLO data.yaml and return (dataset_root, data_cfg, class_names)."""
    data_yaml_path = Path(data_yaml_path)
    data = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))

    dataset_root = Path(data["path"])
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml_path.parent / dataset_root).resolve()

    names = data["names"]
    if isinstance(names, dict):
        class_names = [names[index] for index in sorted(names.keys())]
    else:
        class_names = list(names)

    return dataset_root, data, class_names


def resolve_split_dirs(dataset_root, split_value):
    """Resolve image and label directories from a dataset split value.

    Handles both ``split_value = "train"`` (relative) and absolute paths.
    Automatically detects ``images/`` vs ``labels/`` sub-directories.
    """
    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = (dataset_root / split_path).resolve()

    if split_path.name == "images":
        return split_path, split_path.parent / "labels"

    images_dir = split_path / "images"
    labels_dir = split_path / "labels"

    if images_dir.exists() or labels_dir.exists():
        return images_dir, labels_dir

    return split_path, split_path.parent / "labels"


# ---------------------------------------------------------------------------
# Filename stem helpers
# ---------------------------------------------------------------------------


def canonicalize_stem(stem):
    """Strip trailing ``_NNN`` numeric suffix from a filename stem.

    Example: ``"image_600"`` → ``"image"``
    """
    prefix, separator, suffix = stem.rpartition("_")
    if separator and suffix.isdigit():
        return prefix
    return stem


def extract_size_suffix(stem):
    """Extract trailing numeric suffix from a filename stem, or -1 if absent.

    Example: ``"image_600"`` → ``600``
    """
    prefix, separator, suffix = stem.rpartition("_")
    if separator and suffix.isdigit():
        return int(suffix)
    return -1


# ---------------------------------------------------------------------------
# Label / image file index & lookup
# ---------------------------------------------------------------------------


def _build_index(directory, glob_pattern, filter_fn=None):
    """Build an exact + fallback index for files in *directory*.

    Returns ``(exact_map, fallback_map)`` where *exact_map* maps stem → path
    and *fallback_map* maps canonicalized stem → list of paths (sorted by
    descending size suffix, then name).
    """
    exact_map = {}
    fallback_map = defaultdict(list)

    for file_path in sorted(Path(directory).glob(glob_pattern)):
        if not file_path.is_file():
            continue
        if filter_fn and not filter_fn(file_path):
            continue

        exact_map[file_path.stem] = file_path
        fallback_map[canonicalize_stem(file_path.stem)].append(file_path)

    for candidates in fallback_map.values():
        candidates.sort(
            key=lambda p: (extract_size_suffix(p.stem), p.name),
            reverse=True,
        )

    return exact_map, fallback_map


def build_label_index(labels_dir):
    """Build exact + fallback index for label ``.txt`` files."""
    return _build_index(labels_dir, "*.txt")


def build_image_index(images_dir, image_exts=None):
    """Build exact + fallback index for image files."""
    if image_exts is None:
        image_exts = IMAGE_EXTENSIONS

    def _is_image(p):
        return p.suffix.lower() in image_exts

    return _build_index(images_dir, "*", filter_fn=_is_image)


def find_by_stem(stem, exact_map, fallback_map):
    """Look up a file by *stem* with exact-match then fallback.

    Returns the ``Path`` or ``None``.
    """
    exact_match = exact_map.get(stem)
    if exact_match is not None:
        return exact_match

    candidates = fallback_map.get(canonicalize_stem(stem), [])
    if candidates:
        return candidates[0]

    return None


# ---------------------------------------------------------------------------
# YOLO label parsing
# ---------------------------------------------------------------------------


def yolo_line_to_xyxy(line, image_width, image_height):
    """Convert a YOLO-format label line to ``(class_id, [x1, y1, x2, y2])``
    in pixel coordinates.

    Raises ``ValueError`` if the line has fewer than 5 fields.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"Invalid YOLO label line: {line!r}")

    cls_id = int(float(parts[0]))
    x_center = float(parts[1]) * image_width
    y_center = float(parts[2]) * image_height
    box_width = float(parts[3]) * image_width
    box_height = float(parts[4]) * image_height

    x1 = x_center - box_width / 2
    y1 = y_center - box_height / 2
    x2 = x_center + box_width / 2
    y2 = y_center + box_height / 2
    return cls_id, [x1, y1, x2, y2]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def compute_iou(box_a, box_b):
    """Compute Intersection-over-Union between two ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def add_padding_and_clip(x1, y1, x2, y2, img_w, img_h,
                         padding_ratio=0.25, min_size=32):
    """Add proportional padding to a bbox, enforce minimum size, and clip to
    image boundaries.  Returns ``(x1, y1, x2, y2)`` as integers."""
    box_w = x2 - x1
    box_h = y2 - y1

    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio

    x1 -= pad_x
    y1 -= pad_y
    x2 += pad_x
    y2 += pad_y

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    new_w = max(x2 - x1, min_size)
    new_h = max(y2 - y1, min_size)

    x1 = cx - new_w / 2
    x2 = cx + new_w / 2
    y1 = cy - new_h / 2
    y2 = cy + new_h / 2

    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(img_w, int(round(x2)))
    y2 = min(img_h, int(round(y2)))

    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# Prediction ↔ Ground-truth matching  (Hungarian / optimal)
# ---------------------------------------------------------------------------


def match_predictions(predictions, ground_truths, iou_threshold,
                      class_aware=False):
    """Match predicted boxes to ground-truth boxes using the **Hungarian
    algorithm** (optimal assignment) based on IoU.

    Parameters
    ----------
    predictions : list[dict]
        Each dict must contain ``"bbox"`` ([x1,y1,x2,y2]).  For class-aware
        matching it should also contain ``"stage1_label"`` or ``"label"``.
    ground_truths : list[dict]
        Each dict must contain ``"bbox"`` and ``"label"``.
    iou_threshold : float
        Minimum IoU to accept a match.
    class_aware : bool
        If True, only boxes with matching class labels can be paired.

    Returns
    -------
    matches : list[tuple[int, int, float]]
        ``(pred_idx, gt_idx, iou_value)`` for each valid match.
    unmatched_pred_indices : list[int]
    unmatched_gt_indices : list[int]
    """
    num_preds = len(predictions)
    num_gts = len(ground_truths)

    if num_preds == 0 or num_gts == 0:
        return [], list(range(num_preds)), list(range(num_gts))

    # Build IoU matrix ---------------------------------------------------
    iou_matrix = np.zeros((num_preds, num_gts), dtype=np.float64)
    for p_idx in range(num_preds):
        pred = predictions[p_idx]
        pred_label = pred.get("stage1_label") or pred.get("label", "")
        for g_idx in range(num_gts):
            gt = ground_truths[g_idx]
            if class_aware and pred_label != gt.get("label", ""):
                continue  # leave IoU as 0 → will not be matched
            iou_matrix[p_idx, g_idx] = compute_iou(pred["bbox"], gt["bbox"])

    # Optimal assignment (maximize IoU ⇒ minimize negative IoU) ----------
    from scipy.optimize import linear_sum_assignment

    cost_matrix = -iou_matrix
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches = []
    matched_pred_set = set()
    matched_gt_set = set()

    for pred_idx, gt_idx in zip(row_indices, col_indices):
        iou_value = iou_matrix[pred_idx, gt_idx]
        if iou_value >= iou_threshold:
            matches.append((int(pred_idx), int(gt_idx), float(iou_value)))
            matched_pred_set.add(pred_idx)
            matched_gt_set.add(gt_idx)

    unmatched_pred_indices = [i for i in range(num_preds)
                              if i not in matched_pred_set]
    unmatched_gt_indices = sorted(i for i in range(num_gts)
                                  if i not in matched_gt_set)

    return matches, unmatched_pred_indices, unmatched_gt_indices


# ---------------------------------------------------------------------------
# Image collection
# ---------------------------------------------------------------------------


def collect_image_paths(directory):
    """Collect and sort all image paths from a directory (recursive)."""
    directory = Path(directory)
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
