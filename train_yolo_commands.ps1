# ──────────────────────────────────────────────────────────────
# YOLO Stage 1 Training Commands
# ──────────────────────────────────────────────────────────────
# Run these in PowerShell from the project root (d:\python\PCB).
# Each command trains a different YOLO variant.
# The best model used in the project is v8m_768_adamw_aug (command 3).
# ──────────────────────────────────────────────────────────────

# ── 1) YOLOv8s — baseline, 640px, default SGD optimizer ──────
yolo detect train `
    data=pcb-defect-dataset/data.yaml `
    model=yolov8s.pt `
    epochs=150 `
    imgsz=640 `
    batch=16 `
    device=0 `
    workers=6 `
    patience=40 `
    close_mosaic=10 `
    degrees=5 translate=0.05 scale=0.1 `
    fliplr=0.5 flipud=0.5 `
    hsv_h=0.01 hsv_s=0.3 hsv_v=0.2

# ── 2) YOLOv8s — 768px, AdamW optimizer, cosine LR ──────────
yolo detect train `
    data=pcb-defect-dataset/data.yaml `
    model=yolov8s.pt `
    epochs=150 `
    imgsz=768 `
    batch=16 `
    device=0 `
    workers=6 `
    patience=40 `
    optimizer=AdamW `
    lr0=0.001 lrf=0.01 `
    weight_decay=0.0005 `
    cos_lr=True `
    close_mosaic=10 `
    degrees=5 translate=0.05 scale=0.1 `
    fliplr=0.5 flipud=0.5 `
    hsv_h=0.01 hsv_s=0.3 hsv_v=0.2 `
    amp=True `
    name=v8s_768_adamw_aug

# ── 3) YOLOv8m — 768px, AdamW, cosine LR (BEST MODEL) ──────
#    This produced the final model used in the pipeline:
#    runs/detect/v8m_768_adamw_aug/weights/best.pt
yolo detect train `
    data=pcb-defect-dataset/data.yaml `
    model=yolov8m.pt `
    epochs=150 `
    imgsz=768 `
    batch=16 `
    device=0 `
    workers=6 `
    patience=40 `
    optimizer=AdamW `
    lr0=0.0008 lrf=0.01 `
    weight_decay=0.0005 `
    cos_lr=True `
    close_mosaic=10 `
    degrees=5 translate=0.05 scale=0.1 `
    fliplr=0.5 flipud=0.5 `
    hsv_h=0.01 hsv_s=0.3 hsv_v=0.2 `
    amp=True `
    name=v8m_768_adamw_aug
