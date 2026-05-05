import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm.auto import tqdm

from stage12_yolo_cnn_system import DEFAULT_CNN_PATH, DEFAULT_YOLO_PATH, Stage12Pipeline


def load_dataset_config(data_yaml_path):
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
    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = (dataset_root / split_path).resolve()

    if split_path.name == "images":
        return split_path, split_path.parent / "labels"

    return split_path / "images", split_path / "labels"


def canonicalize_stem(stem):
    prefix, separator, suffix = stem.rpartition("_")
    if separator and suffix.isdigit():
        return prefix
    return stem


def extract_size_suffix(stem):
    prefix, separator, suffix = stem.rpartition("_")
    if separator and suffix.isdigit():
        return int(suffix)
    return -1


def build_label_index(labels_dir):
    exact_map = {}
    fallback_map = defaultdict(list)

    for label_path in sorted(labels_dir.glob("*.txt")):
        exact_map[label_path.stem] = label_path
        fallback_map[canonicalize_stem(label_path.stem)].append(label_path)

    for candidates in fallback_map.values():
        candidates.sort(key=lambda path: (extract_size_suffix(path.stem), path.name), reverse=True)

    return exact_map, fallback_map


def find_label_file(image_stem, exact_map, fallback_map):
    exact_match = exact_map.get(image_stem)
    if exact_match is not None:
        return exact_match

    candidates = fallback_map.get(canonicalize_stem(image_stem), [])
    if candidates:
        return candidates[0]

    return None


def yolo_line_to_xyxy(line, image_width, image_height):
    parts = line.strip().split()
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


def compute_iou(box_a, box_b):
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


def match_predictions(predictions, ground_truths, iou_threshold, class_aware=False):
    matches = []
    unmatched_pred_indices = []
    unmatched_gt_indices = set(range(len(ground_truths)))

    sorted_pred_indices = sorted(
        range(len(predictions)),
        key=lambda index: predictions[index]["stage1_confidence"],
        reverse=True,
    )

    for pred_index in sorted_pred_indices:
        prediction = predictions[pred_index]
        best_gt_index = None
        best_iou = 0.0

        for gt_index in unmatched_gt_indices:
            gt = ground_truths[gt_index]
            if class_aware and prediction["stage1_label"] != gt["label"]:
                continue
            iou = compute_iou(prediction["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if best_gt_index is not None and best_iou >= iou_threshold:
            matches.append((pred_index, best_gt_index, best_iou))
            unmatched_gt_indices.remove(best_gt_index)
        else:
            unmatched_pred_indices.append(pred_index)

    return matches, unmatched_pred_indices, sorted(unmatched_gt_indices)


def run_yolo_val(pipeline, data_yaml, split, imgsz):
    metrics = pipeline.detector.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=16,
        device=pipeline.device_str,
        verbose=False,
        plots=False,
    )

    if hasattr(metrics, "results_dict"):
        results = metrics.results_dict
        return {
            "precision": float(results.get("metrics/precision(B)", 0.0)),
            "recall": float(results.get("metrics/recall(B)", 0.0)),
            "map50": float(results.get("metrics/mAP50(B)", 0.0)),
        }

    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate full YOLO Stage 1 + CNN Stage 2 system")
    parser.add_argument("--data", default="pcb-defect-dataset/data.yaml", help="Path to dataset YAML")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--yolo", default=str(DEFAULT_YOLO_PATH), help="Path to YOLO checkpoint")
    parser.add_argument("--cnn", default=str(DEFAULT_CNN_PATH), help="Path to Stage 2 CNN checkpoint")
    parser.add_argument("--imgsz", type=int, default=768, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold")
    parser.add_argument("--match-iou", type=float, default=0.5, help="IoU threshold for GT/pred matching")
    parser.add_argument("--save-dir", default="runs/system_eval", help="Directory to save evaluation artifacts")
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset_root, data_cfg, class_names = load_dataset_config(args.data)
    images_dir, labels_dir = resolve_split_dirs(dataset_root, data_cfg[args.split])

    pipeline = Stage12Pipeline(
        yolo_path=args.yolo,
        cnn_checkpoint=args.cnn,
        yolo_imgsz=args.imgsz,
        yolo_conf=args.conf,
        yolo_iou=args.iou,
    )

    yolo_metrics = run_yolo_val(pipeline, args.data, args.split, args.imgsz)
    label_exact_map, label_fallback_map = build_label_index(labels_dir)

    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    detection_tp = 0
    detection_fp = 0
    detection_fn = 0
    detection_class_tp = 0
    detection_class_fp = 0
    detection_class_fn = 0
    system_correct = 0
    total_gt = 0
    total_predictions = 0
    classification_y_true = []
    classification_y_pred = []
    pipeline_errors = []
    total_latency_ms = 0.0

    progress = tqdm(image_paths, desc=f"System eval {args.split}", dynamic_ncols=True)

    for image_path in progress:
        start_time = time.perf_counter()
        prediction = pipeline.predict_image(image_path)
        total_latency_ms += (time.perf_counter() - start_time) * 1000.0

        label_path = find_label_file(image_path.stem, label_exact_map, label_fallback_map)
        if label_path is None or not label_path.exists():
            raise FileNotFoundError(f"Missing label file for {image_path.name}")

        image_width = prediction["image_size"]["width"]
        image_height = prediction["image_size"]["height"]

        ground_truths = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cls_id, bbox = yolo_line_to_xyxy(line, image_width, image_height)
            ground_truths.append(
                {
                    "class_id": cls_id,
                    "label": class_names[cls_id],
                    "bbox": bbox,
                }
            )

        matches, unmatched_pred_indices, unmatched_gt_indices = match_predictions(
            prediction["predictions"],
            ground_truths,
            iou_threshold=args.match_iou,
        )
        det_matches, det_unmatched_pred_indices, det_unmatched_gt_indices = match_predictions(
            prediction["predictions"],
            ground_truths,
            iou_threshold=args.match_iou,
            class_aware=True,
        )

        total_gt += len(ground_truths)
        total_predictions += len(prediction["predictions"])
        detection_tp += len(matches)
        detection_fp += len(unmatched_pred_indices)
        detection_fn += len(unmatched_gt_indices)
        detection_class_tp += len(det_matches)
        detection_class_fp += len(det_unmatched_pred_indices)
        detection_class_fn += len(det_unmatched_gt_indices)

        for pred_index, gt_index, iou_value in matches:
            pred = prediction["predictions"][pred_index]
            gt = ground_truths[gt_index]

            classification_y_true.append(gt["label"])
            classification_y_pred.append(pred["stage2_label"])

            if pred["stage2_label"] == gt["label"]:
                system_correct += 1
            else:
                pipeline_errors.append(
                    {
                        "image": image_path.name,
                        "error_type": "misclassification",
                        "gt_label": gt["label"],
                        "pred_label": pred["stage2_label"],
                        "iou": f"{iou_value:.6f}",
                        "stage1_confidence": f"{pred['stage1_confidence']:.6f}",
                        "stage2_confidence": f"{pred['stage2_confidence']:.6f}",
                    }
                )

        for pred_index in unmatched_pred_indices:
            pred = prediction["predictions"][pred_index]
            pipeline_errors.append(
                {
                    "image": image_path.name,
                    "error_type": "false_positive_detection",
                    "gt_label": "",
                    "pred_label": pred["stage2_label"],
                    "iou": "0.000000",
                    "stage1_confidence": f"{pred['stage1_confidence']:.6f}",
                    "stage2_confidence": f"{pred['stage2_confidence']:.6f}",
                }
            )

        for gt_index in unmatched_gt_indices:
            gt = ground_truths[gt_index]
            pipeline_errors.append(
                {
                    "image": image_path.name,
                    "error_type": "missed_detection",
                    "gt_label": gt["label"],
                    "pred_label": "",
                    "iou": "0.000000",
                    "stage1_confidence": "",
                    "stage2_confidence": "",
                }
            )

        running_system_acc = system_correct / total_gt if total_gt else 0.0
        progress.set_postfix(
            det_tp=detection_tp,
            det_fp=detection_fp,
            det_fn=detection_fn,
            system_acc=f"{running_system_acc:.4f}",
        )

    classification_acc = accuracy_score(classification_y_true, classification_y_pred) if classification_y_true else 0.0
    classification_f1 = f1_score(classification_y_true, classification_y_pred, average="macro") if classification_y_true else 0.0
    system_accuracy = system_correct / total_gt if total_gt else 0.0
    det_precision_at_conf = detection_class_tp / (detection_class_tp + detection_class_fp) if (detection_class_tp + detection_class_fp) else 0.0
    det_recall_at_conf = detection_class_tp / (detection_class_tp + detection_class_fn) if (detection_class_tp + detection_class_fn) else 0.0
    avg_latency_ms = total_latency_ms / len(image_paths)

    classification_report_text = classification_report(
        classification_y_true,
        classification_y_pred,
        labels=class_names,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    error_counts = {}
    for item in pipeline_errors:
        error_counts[item["error_type"]] = error_counts.get(item["error_type"], 0) + 1

    summary = {
        "detection": {
            "map50": yolo_metrics["map50"],
            "precision": yolo_metrics["precision"],
            "recall": yolo_metrics["recall"],
            "precision_at_operating_point": det_precision_at_conf,
            "recall_at_operating_point": det_recall_at_conf,
            "tp": detection_class_tp,
            "fp": detection_class_fp,
            "fn": detection_class_fn,
            "tp_localization_only": detection_tp,
            "fp_localization_only": detection_fp,
            "fn_localization_only": detection_fn,
        },
        "classification": {
            "accuracy_on_detected_boxes": classification_acc,
            "macro_f1_on_detected_boxes": classification_f1,
            "num_detected_boxes": len(classification_y_true),
        },
        "system": {
            "total_gt_boxes": total_gt,
            "total_predictions": total_predictions,
            "correct_end_to_end": system_correct,
            "overall_accuracy": system_accuracy,
            "avg_latency_ms_per_image": avg_latency_ms,
            "pipeline_errors": error_counts,
        },
    }

    markdown_report = "\n".join(
        [
            "# YOLO + CNN System Evaluation",
            "",
            f"- Split: `{args.split}`",
            f"- YOLO checkpoint: `{args.yolo}`",
            f"- CNN checkpoint: `{args.cnn}`",
            "",
            "## Detection",
            "",
            f"- mAP50: {summary['detection']['map50']:.4f}",
            f"- Precision: {summary['detection']['precision']:.4f}",
            f"- Recall: {summary['detection']['recall']:.4f}",
            f"- Precision @ current conf: {summary['detection']['precision_at_operating_point']:.4f}",
            f"- Recall @ current conf: {summary['detection']['recall_at_operating_point']:.4f}",
            f"- TP / FP / FN @ current conf: {detection_class_tp} / {detection_class_fp} / {detection_class_fn}",
            f"- Localization-only TP / FP / FN: {detection_tp} / {detection_fp} / {detection_fn}",
            "",
            "## Classification",
            "",
            f"- Accuracy on detected boxes: {classification_acc:.4f}",
            f"- Macro F1 on detected boxes: {classification_f1:.4f}",
            f"- Number of detected boxes evaluated by CNN: {len(classification_y_true)}",
            "",
            "## System",
            "",
            f"- Overall end-to-end accuracy: {system_accuracy:.4f}",
            f"- Correct end-to-end predictions: {system_correct}/{total_gt}",
            f"- Average latency: {avg_latency_ms:.2f} ms/image",
            f"- Pipeline errors: {json.dumps(error_counts, ensure_ascii=False)}",
            "",
            "## Classification Report",
            "",
            "```text",
            classification_report_text,
            "```",
        ]
    )

    print(markdown_report)

    (save_dir / "system_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (save_dir / "system_report.md").write_text(markdown_report + "\n", encoding="utf-8")
    (save_dir / "classification_report.txt").write_text(classification_report_text, encoding="utf-8")

    if pipeline_errors:
        with open(save_dir / "pipeline_errors.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(pipeline_errors[0].keys()))
            writer.writeheader()
            writer.writerows(pipeline_errors)


if __name__ == "__main__":
    main()
