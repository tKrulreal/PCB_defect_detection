from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B2_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    efficientnet_b2,
    resnet18,
    resnet50,
)
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_DROPOUT = 0.30


def build_stage2_model(model_name, num_classes, dropout=DEFAULT_DROPOUT, pretrained=False):
    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.fc.in_features, num_classes),
        )
        return model

    if model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.fc.in_features, num_classes),
        )
        return model

    if model_name == "efficientnet_b2":
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        model = efficientnet_b2(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.classifier[1].in_features, num_classes),
        )
        return model

    raise ValueError(f"Unsupported model_name: {model_name}")


def build_eval_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _checkpoint_num_classes(checkpoint):
    if "idx_to_class" in checkpoint:
        return len(checkpoint["idx_to_class"])
    if "class_to_idx" in checkpoint:
        return len(checkpoint["class_to_idx"])
    raise KeyError("Checkpoint does not contain class mapping.")


def _checkpoint_class_names(checkpoint):
    if "idx_to_class" in checkpoint:
        idx_to_class = checkpoint["idx_to_class"]
        return [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    class_to_idx = checkpoint["class_to_idx"]
    return [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]


def load_stage2_checkpoint(checkpoint_path, device=None):
    checkpoint_path = Path(checkpoint_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint["model_name"]
    image_size = int(checkpoint.get("image_size", 224))
    class_names = _checkpoint_class_names(checkpoint)

    model = build_stage2_model(
        model_name=model_name,
        num_classes=_checkpoint_num_classes(checkpoint),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    transform = build_eval_transform(image_size)
    return {
        "model": model,
        "model_name": model_name,
        "image_size": image_size,
        "class_names": class_names,
        "class_to_idx": {name: idx for idx, name in enumerate(class_names)},
        "transform": transform,
        "checkpoint": checkpoint,
    }


def count_parameters(model, trainable_only=False):
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def format_params(num_params):
    if num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    if num_params >= 1_000:
        return f"{num_params / 1_000:.2f}K"
    return str(num_params)


def prepare_image_tensor(image_rgb, transform, device):
    tensor = transform(image_rgb).unsqueeze(0).to(device)
    if device.type == "cuda":
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    return tensor


def classify_crop_array(image_bgr, bundle, device=None):
    if device is None:
        device = next(bundle["model"].parameters()).device

    image_rgb = Image.fromarray(image_bgr[:, :, ::-1])
    tensor = prepare_image_tensor(image_rgb, bundle["transform"], device)

    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = bundle["model"](tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, pred_idx = probabilities.max(dim=1)

    pred_idx = int(pred_idx.item())
    confidence = float(confidence.item())
    return {
        "pred_idx": pred_idx,
        "pred_label": bundle["class_names"][pred_idx],
        "confidence": confidence,
        "probabilities": probabilities.squeeze(0).cpu().numpy(),
    }


def measure_inference_time(bundle, image_paths, device=None, warmup=20, max_images=200):
    if device is None:
        device = next(bundle["model"].parameters()).device

    image_paths = [Path(path) for path in image_paths[:max_images]]
    if not image_paths:
        raise ValueError("No images provided for inference timing.")

    tensors = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            image_rgb = image.convert("RGB")
        tensor = prepare_image_tensor(image_rgb, bundle["transform"], device)
        tensors.append(tensor)

    warmup_tensors = tensors[: min(warmup, len(tensors))]

    with torch.no_grad():
        for tensor in warmup_tensors:
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                _ = bundle["model"](tensor)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    with torch.no_grad():
        if device.type == "cuda":
            start_event.record()

        if device.type != "cuda":
            import time
            wall_start = time.perf_counter()

        for tensor in tensors:
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                _ = bundle["model"](tensor)

        if device.type == "cuda":
            end_event.record()
            torch.cuda.synchronize(device)
            total_ms = start_event.elapsed_time(end_event)
        else:
            import time
            total_ms = (time.perf_counter() - wall_start) * 1000.0

    return total_ms / len(tensors)
