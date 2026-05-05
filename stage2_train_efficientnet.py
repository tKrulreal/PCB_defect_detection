import copy
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2
from torchvision.transforms import InterpolationMode


MODEL_NAME = "efficientnet_b2"
DATA_DIR = Path("pcb-defect-cls")
OUTPUT_DIR = Path("runs/stage2/efficientnet_b2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_ROOT = OUTPUT_DIR / "tensorboard"
TENSORBOARD_ROOT.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 260
BATCH_SIZE = 48
EPOCHS = 100
BACKBONE_MAX_LR = 2e-4
HEAD_MAX_LR = 8e-4
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.05
DROPOUT = 0.30
NUM_WORKERS = 4
EARLY_STOPPING_PATIENCE = 8
GRAD_CLIP_NORM = 1.0
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"
PIN_MEMORY = DEVICE.type == "cuda"
USE_CHANNELS_LAST = DEVICE.type == "cuda"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_history(history):
    if not history:
        return

    history_path = OUTPUT_DIR / "history.csv"
    fieldnames = list(history[0].keys())

    with open(history_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def prepare_images(images):
    images = images.to(DEVICE, non_blocking=PIN_MEMORY)
    if USE_CHANNELS_LAST:
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def build_summary_writer():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "TensorBoard is not installed. Install it with '.venv\\Scripts\\python.exe -m pip install tensorboard'."
        ) from error

    run_name = time.strftime("%Y%m%d-%H%M%S")
    log_dir = TENSORBOARD_ROOT / run_name
    writer = SummaryWriter(log_dir=str(log_dir))
    return writer, log_dir


def build_dataloaders():
    train_tfms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BILINEAR),
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
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=eval_tfms)
    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=eval_tfms)

    if train_ds.classes != val_ds.classes or train_ds.classes != test_ds.classes:
        raise ValueError("Train/val/test class order does not match.")

    loader_kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": PIN_MEMORY,
    }
    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **loader_kwargs,
    )

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def build_model(num_classes):
    model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(model.classifier[1].in_features, num_classes),
    )
    model = model.to(DEVICE)
    if USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last)
    return model


def build_optimizer(model):
    head_params = list(model.classifier.parameters())
    head_param_ids = {id(param) for param in head_params}
    backbone_params = [
        param for param in model.parameters()
        if id(param) not in head_param_ids
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_MAX_LR},
            {"params": head_params, "lr": HEAD_MAX_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    return optimizer


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, epoch, total_epochs):
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
        images = prepare_images(images)
        labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
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


def evaluate(model, loader, criterion, split_name, epoch=None, total_epochs=None):
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
            images = prepare_images(images)
            labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)

            with torch.amp.autocast(device_type=DEVICE.type, enabled=USE_AMP):
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


def save_checkpoint(path, model_state_dict, class_to_idx, epoch, best_val_acc, best_val_macro_f1):
    torch.save(
        {
            "model_state_dict": model_state_dict,
            "class_to_idx": class_to_idx,
            "idx_to_class": {index: name for name, index in class_to_idx.items()},
            "epoch": epoch,
            "best_val_acc": best_val_acc,
            "best_val_macro_f1": best_val_macro_f1,
            "model_name": MODEL_NAME,
            "image_size": IMAGE_SIZE,
        },
        path,
    )


def main():
    seed_everything(SEED)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print("Device:", DEVICE)

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders()
    num_classes = len(train_ds.classes)
    writer, tensorboard_log_dir = build_summary_writer()

    print("Model:", MODEL_NAME)
    print("Class to idx:", train_ds.class_to_idx)
    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))
    print("Test samples:", len(test_ds))
    print("TensorBoard log dir:", tensorboard_log_dir)
    print("Train class counts:")

    train_counts = np.bincount(train_ds.targets, minlength=num_classes)
    for class_name, class_idx in train_ds.class_to_idx.items():
        print(f"  {class_name}: {int(train_counts[class_idx])}")

    model = build_model(num_classes)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = build_optimizer(model)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[BACKBONE_MAX_LR, HEAD_MAX_LR],
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.15,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1000.0,
    )
    scaler = torch.amp.GradScaler(device=DEVICE.type, enabled=USE_AMP)

    best_metric = -1.0
    best_acc = 0.0
    best_epoch = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history = []

    start = time.time()

    writer.add_text("run/model", MODEL_NAME)
    writer.add_text("run/device", str(DEVICE))
    writer.add_text("run/classes", ", ".join(train_ds.classes))
    writer.add_text(
        "run/hparams",
        "\n".join(
            [
                f"IMAGE_SIZE={IMAGE_SIZE}",
                f"BATCH_SIZE={BATCH_SIZE}",
                f"EPOCHS={EPOCHS}",
                f"BACKBONE_MAX_LR={BACKBONE_MAX_LR}",
                f"HEAD_MAX_LR={HEAD_MAX_LR}",
                f"WEIGHT_DECAY={WEIGHT_DECAY}",
                f"LABEL_SMOOTHING={LABEL_SMOOTHING}",
                f"DROPOUT={DROPOUT}",
                f"NUM_WORKERS={NUM_WORKERS}",
                f"EARLY_STOPPING_PATIENCE={EARLY_STOPPING_PATIENCE}",
            ]
        ),
    )

    try:
        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                total_epochs=EPOCHS,
            )

            val_loss, val_acc, y_val_true, y_val_pred = evaluate(
                model=model,
                loader=val_loader,
                criterion=criterion,
                split_name="val",
                epoch=epoch,
                total_epochs=EPOCHS,
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
            save_history(history)

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("F1/val_macro", val_macro_f1, epoch)
            writer.add_scalar("LR/head", current_lr, epoch)
            writer.add_scalar("LR/backbone", optimizer.param_groups[0]["lr"], epoch)

            print(
                f"Epoch [{epoch:03d}/{EPOCHS}] "
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
                    path=OUTPUT_DIR / "best.pt",
                    model_state_dict=best_model_wts,
                    class_to_idx=train_ds.class_to_idx,
                    epoch=best_epoch,
                    best_val_acc=best_acc,
                    best_val_macro_f1=best_metric,
                )
                print(
                    f"Saved best model: {OUTPUT_DIR / 'best.pt'} "
                    f"(epoch={best_epoch}, val_acc={best_acc:.4f}, val_macro_f1={best_metric:.4f})"
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"after {EARLY_STOPPING_PATIENCE} epochs without improvement."
                )
                break
    finally:
        writer.flush()

    elapsed = time.time() - start
    print(f"Training completed in {elapsed / 60:.2f} minutes")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val acc: {best_acc:.4f}")
    print(f"Best val macro_f1: {best_metric:.4f}")

    save_checkpoint(
        path=OUTPUT_DIR / "last.pt",
        model_state_dict=model.state_dict(),
        class_to_idx=train_ds.class_to_idx,
        epoch=epoch,
        best_val_acc=best_acc,
        best_val_macro_f1=best_metric,
    )

    model.load_state_dict(best_model_wts)
    test_loss, test_acc, y_true, y_pred = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        split_name="test",
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

    (OUTPUT_DIR / "test_classification_report.txt").write_text(report, encoding="utf-8")
    np.savetxt(OUTPUT_DIR / "test_confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    (OUTPUT_DIR / "summary.txt").write_text(
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
