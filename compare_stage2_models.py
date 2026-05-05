import csv
from pathlib import Path

import torch
from torchvision import datasets

from stage2_cnn_utils import count_parameters, format_params, load_stage2_checkpoint, measure_inference_time


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("pcb-defect-cls")
OUTPUT_DIR = Path("runs/stage2/model_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_CONFIGS = [
    ("ResNet18", Path("runs/stage2/resnet18/best.pt"), Path("runs/stage2/resnet18/summary.txt")),
    ("ResNet50", Path("runs/stage2/resnet50/best.pt"), Path("runs/stage2/resnet50/summary.txt")),
    ("EfficientNet", Path("runs/stage2/efficientnet_b2/best.pt"), Path("runs/stage2/efficientnet_b2/summary.txt")),
]


def load_summary(summary_path):
    metrics = {}
    with open(summary_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            metrics[key.strip()] = float(value.strip())
    return metrics


def build_table(rows):
    headers = ["Model", "Val Acc", "Val F1", "Test Acc", "Test F1", "Params", "Inference Time"]
    divider = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(divider) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["Model"],
                    row["Val Acc"],
                    row["Val F1"],
                    row["Test Acc"],
                    row["Test F1"],
                    row["Params"],
                    row["Inference Time"],
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main():
    test_image_paths = sorted(
        path for path in (DATA_DIR / "test").rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not test_image_paths:
        raise FileNotFoundError("No test images found in pcb-defect-cls/test.")

    rows = []
    csv_rows = []

    for display_name, checkpoint_path, summary_path in MODEL_CONFIGS:
        bundle = load_stage2_checkpoint(checkpoint_path, device=DEVICE)
        metrics = load_summary(summary_path)
        params = count_parameters(bundle["model"])
        inference_ms = measure_inference_time(bundle, test_image_paths, device=DEVICE, max_images=200)

        row = {
            "Model": display_name,
            "Val Acc": f"{metrics['best_val_acc']:.4f}",
            "Val F1": f"{metrics['best_val_macro_f1']:.4f}",
            "Test Acc": f"{metrics['test_acc']:.4f}",
            "Test F1": f"{metrics['test_macro_f1']:.4f}",
            "Params": format_params(params),
            "Inference Time": f"{inference_ms:.2f} ms/img",
        }
        rows.append(row)

        csv_rows.append(
            {
                "model": display_name,
                "val_acc": f"{metrics['best_val_acc']:.6f}",
                "val_f1": f"{metrics['best_val_macro_f1']:.6f}",
                "test_acc": f"{metrics['test_acc']:.6f}",
                "test_f1": f"{metrics['test_macro_f1']:.6f}",
                "params": str(params),
                "inference_ms_per_image": f"{inference_ms:.6f}",
                "device": str(DEVICE),
            }
        )

    markdown_table = build_table(rows)
    print(markdown_table)

    (OUTPUT_DIR / "stage2_model_comparison.md").write_text(markdown_table + "\n", encoding="utf-8")

    with open(OUTPUT_DIR / "stage2_model_comparison.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)


if __name__ == "__main__":
    main()
