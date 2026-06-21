"""Crop YOLO-format dataset into per-class image folders for CNN training.

Reads bounding-box annotations from YOLO ``.txt`` label files, crops each
defect region (with padding), and saves the crops into an ImageFolder-compatible
directory structure: ``pcb-defect-cls/{split}/{class_name}/``.
"""

from collections import defaultdict
from pathlib import Path

import cv2
from tqdm import tqdm

from utils import (
    add_padding_and_clip,
    build_image_index,
    find_by_stem,
    load_dataset_config,
    resolve_split_dirs,
    yolo_line_to_xyxy,
)


DATA_YAML = "pcb-defect-dataset/data.yaml"
OUTPUT_DIR = "pcb-defect-cls"
PADDING_RATIO = 0.25
MIN_CROP_SIZE = 32


def crop_split(dataset_root, split_name, split_value, class_names, output_root):
    """Crop all defect regions from a single dataset split."""
    images_dir, labels_dir = resolve_split_dirs(dataset_root, split_value)

    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {labels_dir}")

    image_exact_map, image_fallback_map = build_image_index(images_dir)
    split_counts = defaultdict(int)
    resolve_counts = defaultdict(int)

    for class_name in class_names:
        (output_root / split_name / class_name).mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_dir.glob("*.txt"))

    for label_file in tqdm(label_files, desc=f"Cropping {split_name}"):
        image_file = find_by_stem(label_file.stem, image_exact_map, image_fallback_map)

        if image_file is not None:
            resolve_counts["found"] += 1
        else:
            resolve_counts["missing"] += 1
            print(f"[WARN] Image not found for label: {label_file.name}")
            continue

        img = cv2.imread(str(image_file))
        if img is None:
            print(f"[WARN] Cannot read image: {image_file}")
            continue

        img_h, img_w = img.shape[:2]

        with open(label_file, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file.readlines() if line.strip()]

        for index, line in enumerate(lines):
            try:
                cls_id, bbox = yolo_line_to_xyxy(line, img_w, img_h)
            except ValueError as error:
                print(f"[WARN] {error} in {label_file}")
                continue

            if cls_id < 0 or cls_id >= len(class_names):
                print(f"[WARN] Invalid class id {cls_id} in {label_file}")
                continue

            x1, y1, x2, y2 = add_padding_and_clip(
                bbox[0], bbox[1], bbox[2], bbox[3],
                img_w, img_h,
                padding_ratio=PADDING_RATIO,
                min_size=MIN_CROP_SIZE,
            )

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                print(f"[WARN] Empty crop: {label_file.name}, line {index}")
                continue

            class_name = class_names[cls_id]
            save_name = f"{label_file.stem}_{index}_{class_name}.jpg"
            save_path = output_root / split_name / class_name / save_name

            if not cv2.imwrite(str(save_path), crop):
                print(f"[WARN] Cannot save crop: {save_path}")
                continue

            split_counts[class_name] += 1

    return split_counts, resolve_counts


def main():
    dataset_root, data_cfg, class_names = load_dataset_config(DATA_YAML)
    output_root = Path(OUTPUT_DIR)

    split_map = {
        split_name: data_cfg[split_name]
        for split_name in ("train", "val", "test")
        if data_cfg.get(split_name)
    }
    if not split_map:
        raise KeyError("data.yaml does not define any dataset split.")

    print("Dataset root:", dataset_root)
    print("Output root:", output_root.resolve())
    print("Classes:", class_names)

    total_counts = {}

    for split_name, split_value in split_map.items():
        counts, resolve_counts = crop_split(
            dataset_root, split_name, split_value, class_names, output_root
        )
        total_counts[split_name] = counts

    print("\n===== CROP SUMMARY =====")
    for split_name, counts in total_counts.items():
        print(f"\n[{split_name}]")
        total = 0
        for class_name in class_names:
            count = counts[class_name]
            total += count
            print(f"  {class_name}: {count}")
        print(f"  Total: {total}")


if __name__ == "__main__":
    main()
