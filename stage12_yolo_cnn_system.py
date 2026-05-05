import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from stage2_cnn_utils import classify_crop_array, load_stage2_checkpoint


DEFAULT_YOLO_PATH = Path("runs/detect/v8m_768_adamw_aug/weights/best.pt")
DEFAULT_CNN_PATH = Path("runs/stage2/resnet18/best.pt")


class Stage12Pipeline:
    def __init__(
        self,
        yolo_path=DEFAULT_YOLO_PATH,
        cnn_checkpoint=DEFAULT_CNN_PATH,
        device=None,
        yolo_imgsz=768,
        yolo_conf=0.25,
        yolo_iou=0.7,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_str = "0" if self.device.type == "cuda" else "cpu"
        self.yolo_imgsz = yolo_imgsz
        self.yolo_conf = yolo_conf
        self.yolo_iou = yolo_iou

        self.detector = YOLO(str(yolo_path))
        self.cnn_bundle = load_stage2_checkpoint(cnn_checkpoint, device=self.device)

    @staticmethod
    def _clip_box(x1, y1, x2, y2, width, height):
        x1 = max(0, min(width - 1, int(round(x1))))
        y1 = max(0, min(height - 1, int(round(y1))))
        x2 = max(1, min(width, int(round(x2))))
        y2 = max(1, min(height, int(round(y2))))
        return x1, y1, x2, y2

    def predict_image(self, image_path):
        image_path = Path(image_path)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        height, width = image_bgr.shape[:2]
        results = self.detector.predict(
            source=str(image_path),
            imgsz=self.yolo_imgsz,
            conf=self.yolo_conf,
            iou=self.yolo_iou,
            device=self.device_str,
            verbose=False,
        )
        result = results[0]

        predictions = []
        if result.boxes is None:
            return {
                "image_path": str(image_path),
                "image_size": {"width": width, "height": height},
                "predictions": predictions,
            }

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = self._clip_box(x1, y1, x2, y2, width, height)
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            stage2 = classify_crop_array(crop, self.cnn_bundle, device=self.device)
            stage1_cls_idx = int(box.cls[0].item())
            stage1_conf = float(box.conf[0].item())
            combined_conf = stage1_conf * stage2["confidence"]

            predictions.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "stage1_class_id": stage1_cls_idx,
                    "stage1_label": result.names[stage1_cls_idx],
                    "stage1_confidence": stage1_conf,
                    "stage2_class_id": stage2["pred_idx"],
                    "stage2_label": stage2["pred_label"],
                    "stage2_confidence": stage2["confidence"],
                    "combined_confidence": combined_conf,
                }
            )

        return {
            "image_path": str(image_path),
            "image_size": {"width": width, "height": height},
            "predictions": predictions,
        }


def annotate_predictions(image_path, prediction_result):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    for prediction in prediction_result["predictions"]:
        x1, y1, x2, y2 = prediction["bbox"]
        label = (
            f"{prediction['stage2_label']} "
            f"y:{prediction['stage1_confidence']:.2f} "
            f"c:{prediction['stage2_confidence']:.2f}"
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return image


def collect_input_images(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            path for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        )
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Stage 1 + CNN Stage 2 inference pipeline")
    parser.add_argument("input", help="Input image or directory")
    parser.add_argument("--yolo", default=str(DEFAULT_YOLO_PATH), help="Path to YOLO checkpoint")
    parser.add_argument("--cnn", default=str(DEFAULT_CNN_PATH), help="Path to Stage 2 CNN checkpoint")
    parser.add_argument("--imgsz", type=int, default=768, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold")
    parser.add_argument("--save-dir", default="runs/system_infer", help="Directory to save JSON and annotated images")
    return parser.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Stage12Pipeline(
        yolo_path=args.yolo,
        cnn_checkpoint=args.cnn,
        yolo_imgsz=args.imgsz,
        yolo_conf=args.conf,
        yolo_iou=args.iou,
    )

    image_paths = collect_input_images(args.input)
    all_results = []

    for image_path in image_paths:
        prediction = pipeline.predict_image(image_path)
        all_results.append(prediction)

        annotated = annotate_predictions(image_path, prediction)
        annotated_path = save_dir / f"{Path(image_path).stem}_annotated.jpg"
        json_path = save_dir / f"{Path(image_path).stem}.json"

        cv2.imwrite(str(annotated_path), annotated)
        json_path.write_text(json.dumps(prediction, indent=2), encoding="utf-8")

        print(f"{image_path}: {len(prediction['predictions'])} detections")

    summary_path = save_dir / "predictions_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Saved results to {save_dir}")


if __name__ == "__main__":
    main()
