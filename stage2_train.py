"""Unified Stage-2 CNN training script.

Replaces the three per-model scripts (stage2_train_resnet18.py,
stage2_train_resnet50.py, stage2_train_efficientnet.py) with a single
CLI-driven entry point.

Usage examples
--------------
  python stage2_train.py --model resnet18
  python stage2_train.py --model efficientnet_b2 --epochs 80 --batch-size 32
  python stage2_train.py --model resnet50 --dropout 0.35 --lr-backbone 1e-4
"""

import argparse
import copy
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

from stage2_cnn_utils import build_stage2_model
from utils import seed_everything

# ── ImageNet normalisation constants ────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ── Per-model default hyper-parameters ──────────────────────────────────────
MODEL_DEFAULTS = {
    "resnet18": dict(
        image_size=224,
        batch_size=64,
        epochs=50,
        lr_backbone=3e-4,
        lr_head=1e-3,
        dropout=0.25,
        patience=10,
    ),
    "resnet50": dict(
        image_size=224,
        batch_size=64,
        epochs=100,
        lr_backbone=2e-4,
        lr_head=8e-4,
        dropout=0.30,
        patience=8,
    ),
    "efficientnet_b2": dict(
        image_size=260,
        batch_size=48,
        epochs=100,
        lr_backbone=2e-4,
        lr_head=8e-4,
        dropout=0.30,
        patience=8,
    ),
}

# ── Head attribute name per model family ────────────────────────────────────
HEAD_ATTR = {
    "resnet18": "fc",
    "resnet50": "fc",
    "efficientnet_b2": "classifier",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Stage-2 CNN training (resnet18 / resnet50 / efficientnet_b2)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_DEFAULTS.keys()),
        help="Backbone architecture to train.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="pcb-defect-cls",
        help="Root of the ImageFolder dataset (default: pcb-defect-cls).",
    )

    # Hyper-parameters (None ⇒ will be filled with per-model defaults)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr-backbone", type=float, default=None)
    parser.add_argument("--lr-head", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)

    # Normally-fixed hyper-parameters (still overridable)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Fill None values with per-model defaults
    defaults = MODEL_DEFAULTS[args.model]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    return args


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _device_flags():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    pin_memory = device.type == "cuda"
    use_channels_last = device.type == "cuda"
    return device, use_amp, pin_memory, use_channels_last


def save_history(history, output_dir):
    if not history:
        return

    history_path = output_dir / "history.csv"
    fieldnames = list(history[0].keys())

    with open(history_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def prepare_images(images, device, pin_memory, use_channels_last):
    images = images.to(device, non_blocking=pin_memory)
    if use_channels_last:
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def build_summary_writer(tensorboard_root):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "TensorBoard is not installed. "
            "Install it with '.venv\\Scripts\\python.exe -m pip install tensorboard'."
        ) from error

    run_name = time.strftime("%Y%m%d-%H%M%S")
    log_dir = tensorboard_root / run_name
    writer = SummaryWriter(log_dir=str(log_dir))
    return writer, log_dir


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def build_dataloaders(args, pin_memory):
    image_size = args.image_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    data_dir = Path(args.data_dir)

    train_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=8,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        ),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
        transforms.RandomAutocontrast(p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tfms)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tfms)

    if train_ds.classes != val_ds.classes or train_ds.classes != test_ds.classes:
        raise ValueError("Train/val/test class order does not match.")

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Model & optimiser
# ---------------------------------------------------------------------------


def build_model(args, num_classes, device, use_channels_last):
    model = build_stage2_model(
        model_name=args.model,
        num_classes=num_classes,
        dropout=args.dropout,
        pretrained=True,
    )
    model = model.to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model


def build_optimizer(model, model_name, lr_backbone, lr_head, weight_decay):
    head_attr = HEAD_ATTR[model_name]
    head_params = list(getattr(model, head_attr).parameters())
    head_param_ids = {id(param) for param in head_params}
    backbone_params = [
        param for param in model.parameters()
        if id(param) not in head_param_ids
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )
    return optimizer


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model, loader, criterion, optimizer, scheduler, scaler,
    epoch, total_epochs, device, pin_memory, use_channels_last, use_amp, grad_clip_norm,
):
    model.train()

    running_loss = 0.0
    running_corrects = 0
    total = 0

    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"Train {epoch:03d}/{total_epochs}",
        leave=False,
        dynamic_ncols=True,
    )

    for images, labels in progress:
        images = prepare_images(images, device, pin_memory, use_channels_last)
        labels = labels.to(device, non_blocking=pin_memory)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels).item()
        total += labels.size(0)
        progress.set_postfix(
            loss=f"{running_loss / total:.4f}",
            acc=f"{running_corrects / total:.4f}",
            lr=f"{optimizer.param_groups[-1]['lr']:.2e}",
        )

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc


def evaluate(
    model, loader, criterion, split_name,
    device, pin_memory, use_channels_last, use_amp,
    epoch=None, total_epochs=None,
):
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    total = 0
    all_preds = []
    all_labels = []

    desc = split_name.capitalize()
    if epoch is not None and total_epochs is not None:
        desc = f"{desc} {epoch:03d}/{total_epochs}"

    progress = tqdm(
        loader,
        total=len(loader),
        desc=desc,
        leave=False,
        dynamic_ncols=True,
    )

    with torch.no_grad():
        for images, labels in progress:
            images = prepare_images(images, device, pin_memory, use_channels_last)
            labels = labels.to(device, non_blocking=pin_memory)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            progress.set_postfix(
                loss=f"{running_loss / total:.4f}",
                acc=f"{running_corrects / total:.4f}",
            )

    epoch_loss = running_loss / total
    epoch_acc = running_corrects / total
    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(path, model_state_dict, class_to_idx, epoch,
                    best_val_acc, best_val_macro_f1, model_name, image_size):
    torch.save(
        {
            "model_state_dict": model_state_dict,
            "class_to_idx": class_to_idx,
            "idx_to_class": {index: name for name, index in class_to_idx.items()},
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "best_val_macro_f1": best_val_macro_f1,
            "model_name": model_name,
            "image_size": image_size,
        },
        path,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # ── Setup ───────────────────────────────────────────────────────────────
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device, use_amp, pin_memory, use_channels_last = _device_flags()

    output_dir = Path(f"runs/stage2/{args.model}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_root = output_dir / "tensorboard"
    tensorboard_root.mkdir(parents=True, exist_ok=True)

    print("Device:", device)
    print("Model:", args.model)

    # ── Data ────────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = (
        build_dataloaders(args, pin_memory)
    )
    num_classes = len(train_ds.classes)
    writer, tensorboard_log_dir = build_summary_writer(tensorboard_root)

    print("Class to idx:", train_ds.class_to_idx)
    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))
    print("Test samples:", len(test_ds))
    print("TensorBoard log dir:", tensorboard_log_dir)
    print("Train class counts:")

    train_counts = np.bincount(train_ds.targets, minlength=num_classes)
    for class_name, class_idx in train_ds.class_to_idx.items():
        print(f"  {class_name}: {int(train_counts[class_idx])}")

    # ── Model / Optimiser / Scheduler / Scaler ──────────────────────────────
    model = build_model(args, num_classes, device, use_channels_last)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(
        model, args.model, args.lr_backbone, args.lr_head, args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lr_backbone, args.lr_head],
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.15,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1000.0,
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    # ── Training state ──────────────────────────────────────────────────────
    best_metric = -1.0
    best_acc = 0.0
    best_epoch = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []

    start = time.time()

    writer.add_text("run/model", args.model)
    writer.add_text("run/device", str(device))
    writer.add_text("run/classes", ", ".join(train_ds.classes))
    writer.add_text(
        "run/hparams",
        "\n".join(
            [
                f"IMAGE_SIZE={args.image_size}",
                f"BATCH_SIZE={args.batch_size}",
                f"EPOCHS={args.epochs}",
                f"BACKBONE_MAX_LR={args.lr_backbone}",
                f"HEAD_MAX_LR={args.lr_head}",
                f"WEIGHT_DECAY={args.weight_decay}",
                f"LABEL_SMOOTHING={args.label_smoothing}",
                f"DROPOUT={args.dropout}",
                f"NUM_WORKERS={args.num_workers}",
                f"EARLY_STOPPING_PATIENCE={args.patience}",
            ]
        ),
    )

    # ── Training loop ───────────────────────────────────────────────────────
    epoch = 0  # ensure defined even if EPOCHS == 0
    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                total_epochs=args.epochs,
                device=device,
                pin_memory=pin_memory,
                use_channels_last=use_channels_last,
                use_amp=use_amp,
                grad_clip_norm=args.grad_clip_norm,
            )

            val_loss, val_acc, y_val_true, y_val_pred = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                split_name="val",
                device=device,
                pin_memory=pin_memory,
                use_channels_last=use_channels_last,
                use_amp=use_amp,
                epoch=epoch,
                total_epochs=args.epochs,
            )
            val_macro_f1 = f1_score(y_val_true, y_val_pred, average="macro")
            current_lr = optimizer.param_groups[-1]["lr"]

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "train_acc": f"{train_acc:.6f}",
                    "val_loss": f"{val_loss:.6f}",
                    "val_acc": f"{val_acc:.6f}",
                    "val_macro_f1": f"{val_macro_f1:.6f}",
                    "lr": f"{current_lr:.8f}",
                }
            )
            save_history(history, output_dir)

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("F1/val_macro", val_macro_f1, epoch)
            writer.add_scalar("LR/head", current_lr, epoch)
            writer.add_scalar("LR/backbone", optimizer.param_groups[0]["lr"], epoch)

            print(
                f"Epoch [{epoch:03d}/{args.epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"val_macro_f1={val_macro_f1:.4f} lr={current_lr:.6f}"
            )

            is_better = (
                val_macro_f1 > best_metric
                or (abs(val_macro_f1 - best_metric) < 1e-8 and val_acc > best_acc)
            )

            if is_better:
                best_metric = val_macro_f1
                best_acc = val_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0

                save_checkpoint(
                    path=output_dir / "best.pt",
                    model_state_dict=best_model_wts,
                    class_to_idx=train_ds.class_to_idx,
                    epoch=best_epoch,
                    best_val_acc=best_acc,
                    best_val_macro_f1=best_metric,
                    model_name=args.model,
                    image_size=args.image_size,
                )
                print(
                    f"Saved best model: {output_dir / 'best.pt'} "
                    f"(epoch={best_epoch}, val_acc={best_acc:.4f}, val_macro_f1={best_metric:.4f})"
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= args.patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"after {args.patience} epochs without improvement."
                )
                break
    finally:
        writer.flush()

    elapsed = time.time() - start
    print(f"Training completed in {elapsed / 60:.2f} minutes")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val acc: {best_acc:.4f}")
    print(f"Best val macro_f1: {best_metric:.4f}")

    # ── Save last checkpoint ────────────────────────────────────────────────
    save_checkpoint(
        path=output_dir / "last.pt",
        model_state_dict=model.state_dict(),
        class_to_idx=train_ds.class_to_idx,
        epoch=epoch,
        best_val_acc=best_acc,
        best_val_macro_f1=best_metric,
        model_name=args.model,
        image_size=args.image_size,
    )

    # ── Test evaluation ─────────────────────────────────────────────────────
    model.load_state_dict(best_model_wts)
    test_loss, test_acc, y_true, y_pred = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        split_name="test",
        device=device,
        pin_memory=pin_memory,
        use_channels_last=use_channels_last,
        use_amp=use_amp,
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=train_ds.classes,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)
    test_macro_f1 = f1_score(y_true, y_pred, average="macro")

    print("\n===== TEST RESULT =====")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test acc:  {test_acc:.4f}")
    print(f"Test macro_f1: {test_macro_f1:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)

    (output_dir / "test_classification_report.txt").write_text(report, encoding="utf-8")
    np.savetxt(output_dir / "test_confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    (output_dir / "summary.txt").write_text(
        "\n".join(
            [
                f"best_epoch={best_epoch}",
                f"best_val_acc={best_acc:.6f}",
                f"best_val_macro_f1={best_metric:.6f}",
                f"test_loss={test_loss:.6f}",
                f"test_acc={test_acc:.6f}",
                f"test_macro_f1={test_macro_f1:.6f}",
            ]
        ),
        encoding="utf-8",
    )

    writer.add_scalar("Loss/test", test_loss, best_epoch)
    writer.add_scalar("Accuracy/test", test_acc, best_epoch)
    writer.add_scalar("F1/test_macro", test_macro_f1, best_epoch)
    writer.add_text("test/classification_report", report)
    writer.close()


if __name__ == "__main__":
    main()
