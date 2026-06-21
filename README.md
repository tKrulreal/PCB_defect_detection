# PCB Defect Detection — YOLO + CNN (Stage 1 + Stage 2)

Hệ thống phát hiện và phân loại lỗi PCB sử dụng pipeline 2 giai đoạn:
- **Stage 1 (YOLO):** phát hiện bounding box vùng lỗi trên bo mạch.
- **Stage 2 (CNN):** phân loại loại lỗi cho từng bounding box.

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Web Application](#web-application)
- [Testing](#testing)
- [Results](#results)
- [TensorBoard](#tensorboard)
- [Output Folders](#output-folders)

## Pipeline Overview

```
┌─────────────┐     crop + pad     ┌─────────────┐     aggregate     ┌──────────┐
│  PCB Image  │ ──► YOLO Stage 1 ──► CNN Stage 2 │ ──────────────► │  Output  │
│             │     detect bbox    │  classify    │                  │ JSON+IMG │
└─────────────┘                    └─────────────┘                  └──────────┘
```

1. **YOLO** phát hiện bounding box trên ảnh PCB.
2. Crop ảnh theo bounding box (thêm padding 25%, đảm bảo kích thước tối thiểu 32px).
3. **CNN** phân loại loại lỗi và tính confidence score.
4. Tổng hợp kết quả → lưu JSON + ảnh annotated.

## Project Structure

### Core Pipeline

| File | Mô tả |
| --- | --- |
| `stage12_yolo_cnn_system.py` | Pipeline YOLO + CNN inference, xuất JSON + ảnh annotated |
| `stage2_cnn_utils.py` | Tiện ích CNN: load checkpoint, preprocess, classify crop |
| `utils.py` | **Module tiện ích chung** — gom các hàm dùng chung (IoU, YOLO parsing, Hungarian matching, ...) |

### Training

| File | Mô tả |
| --- | --- |
| `stage2_train.py` | **Script training hợp nhất** — thay thế 3 file riêng lẻ, chọn model qua `--model` |
| `stage2_crop_dataset.py` | Crop từ YOLO dataset → CNN dataset (ImageFolder format) |
| `train_yolo_commands.ps1` | Các lệnh training YOLO Stage 1 (PowerShell) |

### Evaluation & Visualization

| File | Mô tả |
| --- | --- |
| `evaluate_stage12_system.py` | Đánh giá end-to-end (mAP/precision/recall + accuracy + latency), sử dụng **Hungarian matching** |
| `compare_stage2_models.py` | So sánh ResNet18/50/EfficientNet và xuất bảng |
| `visualize_results.py` | **Sinh confusion matrix heatmap + training curves** từ kết quả training |
| `gradcam_visualize.py` | **GradCAM** — hiển thị vùng ảnh CNN tập trung khi phân loại |

### Demo & Web App

| File | Mô tả |
| --- | --- |
| `demo_stage12_one_image.py` | Quick demo — thả ảnh vào `demo_input/` |
| `app.py` | **Streamlit web application** — upload ảnh, xem kết quả trực quan trên trình duyệt |
| `main.py` | **CLI entry point** — điểm vào duy nhất cho mọi chức năng |

### Testing & Config

| File | Mô tả |
| --- | --- |
| `tests/test_utils.py` | 46 unit tests cho các hàm tiện ích |
| `tests/test_pipeline.py` | 12 integration tests cho pipeline |
| `requirements.txt` | Dependencies với version tối thiểu |
| `.gitignore` | Ignore rules cho Git |

### Data Folders

| Folder | Nội dung |
| --- | --- |
| `pcb-defect-dataset/` | YOLO dataset (images/labels theo split) |
| `pcb-defect-cls/` | CNN dataset (images phân loại theo type) |
| `runs/` | Output của training/evaluation/inference |
| `demo_input/` | Ảnh đầu vào cho quick demo |
| `demo_output/` | Kết quả demo |

## Dataset

- **Nguồn:** [PCB Defect Dataset — Kaggle](https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset)
- **6 loại lỗi:** `missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`

## Installation

```powershell
# Clone project
git clone <repository-url>
cd PCB

# Tạo virtual environment
python -m venv .venv
.venv\Scripts\activate

# Cài dependencies
pip install -r requirements.txt
```

## Quick Start

### CLI Entry Point

Tất cả chức năng đều truy cập qua `main.py`:

```powershell
python main.py demo          # Quick demo
python main.py train         # Train CNN Stage 2
python main.py evaluate      # Đánh giá pipeline
python main.py visualize     # Sinh hình visualization
python main.py gradcam       # GradCAM heatmap
python main.py compare       # So sánh models
python main.py app           # Mở web app
```

### Quick Demo

1. Đặt ảnh vào thư mục `demo_input/`.
2. Chạy:

```powershell
python main.py demo
```

3. Kết quả lưu trong `demo_output/`:
   - `*_annotated.jpg`: Ảnh với bounding box và nhãn
   - `*.json`: Chi tiết predictions
   - `predictions_summary.json`: Tóm tắt

### Demo với ảnh cụ thể

```powershell
python main.py demo --input D:\path\to\image.jpg --yolo runs\detect\v8m_768_adamw_aug\weights\best.pt --cnn runs\stage2\resnet18\best.pt
```

### Full Inference

```powershell
python stage12_yolo_cnn_system.py D:\path\to\images --yolo runs\detect\v8m_768_adamw_aug\weights\best.pt --cnn runs\stage2\resnet18\best.pt --imgsz 768 --conf 0.25 --iou 0.7 --save-dir runs\system_infer
```

## Training

### Stage 1 — YOLO

Xem file `train_yolo_commands.ps1` để biết các lệnh training YOLO. Model tốt nhất hiện tại:

```powershell
yolo detect train data=pcb-defect-dataset/data.yaml model=yolov8m.pt epochs=150 imgsz=768 batch=16 optimizer=AdamW lr0=0.0008 cos_lr=True name=v8m_768_adamw_aug
```

### Stage 2 — CNN

Script training hợp nhất hỗ trợ 3 architectures:

```powershell
# Train từng model với default hyperparameters
python main.py train --model resnet18
python main.py train --model resnet50
python main.py train --model efficientnet_b2

# Tùy chỉnh hyperparameters
python main.py train --model resnet18 --epochs 30 --batch-size 32 --dropout 0.35
python main.py train --model efficientnet_b2 --lr-backbone 1e-4 --patience 15
```

**Default hyperparameters:**

| Tham số | ResNet18 | ResNet50 | EfficientNet-B2 |
| --- | --- | --- | --- |
| Image Size | 224 | 224 | 260 |
| Batch Size | 64 | 64 | 48 |
| Epochs | 50 | 100 | 100 |
| Backbone LR | 3e-4 | 2e-4 | 2e-4 |
| Head LR | 1e-3 | 8e-4 | 8e-4 |
| Dropout | 0.25 | 0.30 | 0.30 |
| Patience | 10 | 8 | 8 |

**Các kỹ thuật training:**
- AdamW optimizer với differential learning rate (backbone vs head)
- OneCycleLR scheduler (15% warmup, cosine annealing)
- Mixed Precision Training (AMP)
- Gradient clipping (max norm = 1.0)
- Label smoothing (ε = 0.05)
- Early stopping theo validation macro-F1

**Output:**
- `runs/stage2/{model}/best.pt` — checkpoint tốt nhất
- `runs/stage2/{model}/last.pt` — checkpoint cuối
- `runs/stage2/{model}/history.csv` — lịch sử training
- `runs/stage2/{model}/test_confusion_matrix.csv` — confusion matrix
- `runs/stage2/{model}/test_classification_report.txt` — classification report

### Chuẩn bị dữ liệu CNN

```powershell
python stage2_crop_dataset.py
```

Crop từ YOLO dataset (`pcb-defect-dataset/`) → CNN dataset (`pcb-defect-cls/`) theo cấu trúc ImageFolder.

## Evaluation

### End-to-End System Evaluation

Sử dụng **Hungarian matching** (optimal assignment) để ghép predictions với ground truth:

```powershell
python main.py evaluate --data pcb-defect-dataset\data.yaml --split test --yolo runs\detect\v8m_768_adamw_aug\weights\best.pt --cnn runs\stage2\resnet18\best.pt --imgsz 768 --conf 0.25 --iou 0.7 --match-iou 0.5 --save-dir runs\system_eval\resnet18_test
```

**Output:**
- `system_summary.json` — metrics dạng JSON
- `system_report.md` — báo cáo Markdown
- `classification_report.txt` — per-class metrics
- `pipeline_errors.csv` — danh sách lỗi chi tiết

## Visualization

### Confusion Matrix + Training Curves

Tự động sinh hình từ kết quả training:

```powershell
python main.py visualize --model all        # Tất cả models
python main.py visualize --model resnet18   # Một model cụ thể
```

**Output cho mỗi model:**
- `runs/stage2/{model}/confusion_matrix.png` — heatmap với số lượng + phần trăm
- `runs/stage2/{model}/training_curves.png` — 4 biểu đồ (loss, accuracy, F1, learning rate)

### GradCAM

Hiển thị vùng ảnh CNN tập trung khi phân loại — giúp giải thích quyết định của AI:

```powershell
# Một ảnh
python main.py gradcam --input demo_input/image.jpg --cnn runs/stage2/resnet18/best.pt

# Thư mục ảnh (tối đa 10)
python main.py gradcam --input pcb-defect-cls/test/mouse_bite/ --cnn runs/stage2/resnet18/best.pt --max-images 10
```

**Output:** `runs/gradcam/` — ảnh 3 panel (Original | Heatmap | Overlay)

## Web Application

Giao diện web Streamlit cho phép upload ảnh PCB và xem kết quả trực quan:

```powershell
python main.py app
# hoặc
streamlit run app.py
```

Mở trình duyệt → `http://localhost:8501`

**Tính năng:**
- Upload nhiều ảnh cùng lúc (JPG, PNG, BMP)
- Chọn CNN model (ResNet18, ResNet50, EfficientNet-B2)
- Điều chỉnh confidence/IoU threshold
- Xem ảnh gốc vs ảnh annotated song song
- Bảng chi tiết từng defect với color-coded badges
- Summary dashboard: metric cards + bar chart
- Download kết quả JSON

## Testing

```powershell
# Chạy tất cả tests
python -m pytest tests/ -v

# Chạy từng file
python -m pytest tests/test_utils.py -v       # 46 unit tests
python -m pytest tests/test_pipeline.py -v    # 12 integration tests

# Chạy tests theo keyword
python -m pytest tests/ -v -k "iou"           # Chỉ tests liên quan IoU
```

**Test coverage:**
- `test_utils.py` (46 tests): IoU, YOLO parsing, stem helpers, padding, Hungarian matching
- `test_pipeline.py` (12 tests): annotate predictions, collect images, CNN classify

## Results

### End-to-End Results Summary (test split)

Source: `runs/system_eval/*/system_summary.json`

| Stage 2 Model | System Accuracy | Classification Accuracy (on detected) | Macro F1 Score | Latency (ms/img) | Misclassifications | Missed Defects | False Positives |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 | 0.7484 | 0.9770 | 0.9777 | 33.05 | 38 | 505 | 36 |
| ResNet50 | 0.7400 | 0.9661 | 0.9662 | 37.54 | 56 | 505 | 36 |
| EfficientNet-B2 | 0.7567 | 0.9879 | 0.9880 | 51.38 | 20 | 505 | 36 |

> **Ghi chú:** Bottleneck chính là Stage 1 (YOLO) bỏ sót ~505 defects trên test set. Stage 2 CNN phân loại rất tốt (>97% accuracy trên các box đã phát hiện).

### Stage 2 Model Comparison

Source: `runs/stage2/model_comparison/stage2_model_comparison.md`

| Model | Val Accuracy | Val F1 Score | Test Accuracy | Test F1 Score | Parameters | Inference Time |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet18 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 11.18M | 2.89 ms/img |
| ResNet50 | 0.9995 | 0.9995 | 0.9991 | 0.9990 | 23.52M | 6.58 ms/img |
| EfficientNet-B2 | 0.9995 | 0.9995 | 0.9981 | 0.9981 | 7.71M | 14.55 ms/img |

### Example Demo Output

![Demo annotated](demo_output/l_light_01_missing_hole_18_3_600_annotated.jpg)

### Stage 2 Confusion Matrices

| ResNet-18 | ResNet-50 | EfficientNet-B2 |
| --- | --- | --- |
| ![ResNet18 CM](runs/stage2/resnet18/confusion_matrix.png) | ![ResNet50 CM](runs/stage2/resnet50/confusion_matrix.png) | ![EfficientNet CM](runs/stage2/efficientnet_b2/confusion_matrix.png) |

### Stage 2 Training Curves

| ResNet-18 | ResNet-50 | EfficientNet-B2 |
| --- | --- | --- |
| ![ResNet18 curves](runs/stage2/resnet18/training_curves.png) | ![ResNet50 curves](runs/stage2/resnet50/training_curves.png) | ![EfficientNet curves](runs/stage2/efficientnet_b2/training_curves.png) |

### YOLO Metrics (Stage 1)

![YOLO results](runs/detect/v8m_768_adamw_aug/results.png)
![YOLO PR curve](runs/detect/v8m_768_adamw_aug/BoxPR_curve.png)
![YOLO F1 curve](runs/detect/v8m_768_adamw_aug/BoxF1_curve.png)
![YOLO Precision curve](runs/detect/v8m_768_adamw_aug/BoxP_curve.png)
![YOLO Recall curve](runs/detect/v8m_768_adamw_aug/BoxR_curve.png)
![YOLO Confusion Matrix](runs/detect/v8m_768_adamw_aug/confusion_matrix.png)
![YOLO Confusion Matrix Normalized](runs/detect/v8m_768_adamw_aug/confusion_matrix_normalized.png)

### YOLO Prediction Samples

![Val batch pred 0](runs/detect/v8m_768_adamw_aug/val_batch0_pred.jpg)
![Val batch pred 1](runs/detect/v8m_768_adamw_aug/val_batch1_pred.jpg)
![Val batch pred 2](runs/detect/v8m_768_adamw_aug/val_batch2_pred.jpg)

## TensorBoard

TensorBoard logs cho Stage 2 được lưu tại:
- `runs/stage2/resnet18/tensorboard/`
- `runs/stage2/resnet50/tensorboard/`
- `runs/stage2/efficientnet_b2/tensorboard/`

```powershell
tensorboard --logdir runs\stage2
```

Mở trình duyệt → `http://localhost:6006`

![TensorBoard overview](docs/tensorboard_overview.png)

## Output Folders

| Folder | Nội dung |
| --- | --- |
| `runs/detect/v8m_768_adamw_aug/` | YOLO training output + metric images |
| `runs/stage2/{model}/` | CNN checkpoints, confusion matrix, training curves, TensorBoard logs |
| `runs/system_infer/` | YOLO + CNN inference output |
| `runs/system_eval/*/` | End-to-end evaluation reports + error CSV |
| `runs/gradcam/` | GradCAM heatmap visualizations |
| `demo_output/` | Quick demo output |
