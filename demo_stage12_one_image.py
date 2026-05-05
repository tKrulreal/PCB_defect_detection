import argparse
import json
from pathlib import Path

import cv2

from stage12_yolo_cnn_system import (
    DEFAULT_CNN_PATH,
    DEFAULT_YOLO_PATH,
    Stage12Pipeline,
    annotate_predictions,
    collect_input_images,
)


DEFAULT_INPUT_DIR = Path("demo_input")
DEFAULT_OUTPUT_DIR = Path("demo_output")


def parse_args():
    parser = argparse.ArgumentParser(description="Demo: run YOLO + CNN pipeline on input images")
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_DIR),
        help="Input image or directory (default: demo_input)",
    )
    parser.add_argument(
        "--save-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for annotated images and JSON (default: demo_output)",
    )
    parser.add_argument("--yolo", default=str(DEFAULT_YOLO_PATH), help="Path to YOLO checkpoint")
    parser.add_argument("--cnn", default=str(DEFAULT_CNN_PATH), help="Path to Stage 2 CNN checkpoint")
    parser.add_argument("--imgsz", type=int, default=768, help="YOLO inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold")
    return parser.parse_args()


def ensure_input_path(input_path):
    if input_path.exists():
        return
    input_path.mkdir(parents=True, exist_ok=True)
    raise FileNotFoundError(
        f"Created {input_path} folder. Drop images there and re-run the script."
    )


def main():
    args = parse_args()
    input_path = Path(args.input)
    ensure_input_path(input_path)

    image_paths = collect_input_images(input_path)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_path}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Stage12Pipeline(
        yolo_path=args.yolo,
        cnn_checkpoint=args.cnn,
        yolo_imgsz=args.imgsz,
        yolo_conf=args.conf,
        yolo_iou=args.iou,
    )

    all_results = []
    for image_path in image_paths:
        prediction = pipeline.predict_image(image_path)
        all_results.append(prediction)

        annotated = annotate_predictions(image_path, prediction)
        annotated_path = save_dir / f"{image_path.stem}_annotated.jpg"
        json_path = save_dir / f"{image_path.stem}.json"

        cv2.imwrite(str(annotated_path), annotated)
        json_path.write_text(json.dumps(prediction, indent=2), encoding="utf-8")

        print(f"{image_path}: {len(prediction['predictions'])} detections")

    summary_path = save_dir / "predictions_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Saved results to {save_dir}")


if __name__ == "__main__":
    main()

