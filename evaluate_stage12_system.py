"""End-to-end evaluation of the YOLO Stage 1 + CNN Stage 2 pipeline.

Runs the full inference pipeline on a dataset split, matches predictions to
ground-truth annotations, and computes detection + classification metrics.
Uses **Hungarian (optimal) matching** for prediction-to-GT assignment.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm.auto import tqdm

from stage12_yolo_cnn_system import DEFAULT_CNN_PATH, DEFAULT_YOLO_PATH, Stage12Pipeline
from utils import (
    build_label_index,
    find_by_stem,
    load_dataset_config,
    match_predictions,
    resolve_split_dirs,
    yolo_line_to_xyxy,
)


def run_yolo_val(pipeline, data_yaml, split, imgsz):
    """Run YOLO built-in validation and return mAP/precision/recall dict."""
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
    parser = argparse.ArgumentParser(
        description="Evaluate full YOLO Stage 1 + CNN Stage 2 system"
    )
    parser.add_argument(
        "--data", default="pcb-defect-dataset/data.yaml",
        help="Path to dataset YAML",
    )
    parser.add_argument(
        "--split", default="test", choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--yolo", default=str(DEFAULT_YOLO_PATH),
        help="Path to YOLO checkpoint",
    )
    parser.add_argument(
        "--cnn", default=str(DEFAULT_CNN_PATH),
        help="Path to Stage 2 CNN checkpoint",
    )
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

    # ---- Accumulators ----
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

        label_path = find_by_stem(image_path.stem, label_exact_map, label_fallback_map)
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

        # Localization-only matching (class-agnostic)
        matches, unmatched_pred_indices, unmatched_gt_indices = match_predictions(
            prediction["predictions"],
            ground_truths,
            iou_threshold=args.match_iou,
        )
        # Class-aware matching
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

    # ---- Compute final metrics ----
    classification_acc = (
        accuracy_score(classification_y_true, classification_y_pred)
        if classification_y_true else 0.0
    )
    classification_f1 = (
        f1_score(classification_y_true, classification_y_pred, average="macro")
        if classification_y_true else 0.0
    )
    system_accuracy = system_correct / total_gt if total_gt else 0.0
    det_precision_at_conf = (
        detection_class_tp / (detection_class_tp + detection_class_fp)
        if (detection_class_tp + detection_class_fp) else 0.0
    )
    det_recall_at_conf = (
        detection_class_tp / (detection_class_tp + detection_class_fn)
        if (detection_class_tp + detection_class_fn) else 0.0
    )
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
            f"- Matching algorithm: **Hungarian (optimal)**",
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

    (save_dir / "system_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (save_dir / "system_report.md").write_text(
        markdown_report + "\n", encoding="utf-8"
    )
    (save_dir / "classification_report.txt").write_text(
        classification_report_text, encoding="utf-8"
    )

    if pipeline_errors:
        with open(save_dir / "pipeline_errors.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(pipeline_errors[0].keys()))
            writer.writeheader()
            writer.writerows(pipeline_errors)


if __name__ == "__main__":
    main()
