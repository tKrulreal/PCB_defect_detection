from collections import defaultdict
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm


DATA_YAML = "pcb-defect-dataset/data.yaml"
OUTPUT_DIR = "pcb-defect-cls"
PADDING_RATIO = 0.25
MIN_CROP_SIZE = 32

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_yaml(data_yaml_path):
    data_yaml_path = Path(data_yaml_path)
    with open(data_yaml_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    dataset_root = Path(data["path"])
    if not dataset_root.is_absolute():
        dataset_root = (data_yaml_path.parent / dataset_root).resolve()

    split_map = {
        split_name: data[split_name]
        for split_name in ("train", "val", "test")
        if data.get(split_name)
    }
    if not split_map:
        raise KeyError("data.yaml does not define any dataset split.")

    names = data["names"]
    if isinstance(names, dict):
        class_names = [names[index] for index in sorted(names.keys())]
    else:
        class_names = list(names)

    return dataset_root, split_map, class_names


def resolve_split_dirs(dataset_root, split_value):
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


def yolo_to_xyxy(line, img_w, img_h):
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"Invalid YOLO label line: {line!r}")

    cls_id = int(float(parts[0]))
    x_center = float(parts[1]) * img_w
    y_center = float(parts[2]) * img_h
    box_w = float(parts[3]) * img_w
    box_h = float(parts[4]) * img_h

    x1 = x_center - box_w / 2
    y1 = y_center - box_h / 2
    x2 = x_center + box_w / 2
    y2 = y_center + box_h / 2

    return cls_id, x1, y1, x2, y2


def add_padding_and_clip(x1, y1, x2, y2, img_w, img_h, padding_ratio=0.25, min_size=32):
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


def build_image_index(images_dir):
    exact_map = {}
    fallback_map = defaultdict(list)

    for image_file in sorted(images_dir.iterdir()):
        if not image_file.is_file():
            continue
        if image_file.suffix.lower() not in IMAGE_EXTS:
            continue

        exact_map[image_file.stem] = image_file
        fallback_map[canonicalize_stem(image_file.stem)].append(image_file)

    for candidates in fallback_map.values():
        candidates.sort(key=lambda path: (extract_size_suffix(path.stem), path.name), reverse=True)

    return exact_map, fallback_map


def find_image_file(label_stem, exact_map, fallback_map):
    exact_match = exact_map.get(label_stem)
    if exact_match is not None:
        return exact_match, "exact"

    candidates = fallback_map.get(canonicalize_stem(label_stem), [])
    if candidates:
        return candidates[0], "fallback"

    return None, "missing"


def crop_split(dataset_root, split_name, split_value, class_names, output_root):
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
        image_file, match_type = find_image_file(label_file.stem, image_exact_map, image_fallback_map)
        resolve_counts[match_type] += 1

        if image_file is None:
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
                cls_id, x1, y1, x2, y2 = yolo_to_xyxy(line, img_w, img_h)
            except ValueError as error:
                print(f"[WARN] {error} in {label_file}")
                continue

            if cls_id < 0 or cls_id >= len(class_names):
                print(f"[WARN] Invalid class id {cls_id} in {label_file}")
                continue

            x1, y1, x2, y2 = add_padding_and_clip(
                x1,
                y1,
                x2,
                y2,
                img_w,
                img_h,
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
    dataset_root, split_map, class_names = load_yaml(DATA_YAML)
    output_root = Path(OUTPUT_DIR)

    print("Dataset root:", dataset_root)
    print("Output root:", output_root.resolve())
    print("Classes:", class_names)

    total_counts = {}
    total_resolve_counts = {}

    for split_name, split_value in split_map.items():
        counts, resolve_counts = crop_split(dataset_root, split_name, split_value, class_names, output_root)
        total_counts[split_name] = counts
        total_resolve_counts[split_name] = resolve_counts

    print("\n===== CROP SUMMARY =====")
    for split_name, counts in total_counts.items():
        print(f"\n[{split_name}]")
        total = 0
        for class_name in class_names:
            count = counts[class_name]
            total += count
            print(f"{class_name}: {count}")
        print(f"Total: {total}")

        resolve_counts = total_resolve_counts[split_name]
        print(
            "Image matching:"
            f" exact={resolve_counts['exact']},"
            f" fallback={resolve_counts['fallback']},"
            f" missing={resolve_counts['missing']}"
        )


if __name__ == "__main__":
    main()
