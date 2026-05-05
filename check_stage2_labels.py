from pathlib import Path
from collections import Counter, defaultdict

DATASET_DIR = Path("pcb-defect-dataset")

for split in ["train", "val", "test"]:
    labels_dir = DATASET_DIR / split / "labels"

    total_lines = 0
    total_unique_lines = 0
    duplicate_files = 0
    class_counts = Counter()

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        unique_lines = list(dict.fromkeys(lines))

        total_lines += len(lines)
        total_unique_lines += len(unique_lines)

        if len(lines) != len(unique_lines):
            duplicate_files += 1

        for line in unique_lines:
            cls_id = int(float(line.split()[0]))
            class_counts[cls_id] += 1

    print(f"\n[{split}]")
    print("Raw label lines:", total_lines)
    print("Unique label lines:", total_unique_lines)
    print("Files with duplicate labels:", duplicate_files)
    print("Class counts:", dict(class_counts))