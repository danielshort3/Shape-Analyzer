"""Train/serve preprocessing contract for newly exported state dictionaries."""
import hashlib
import json
from functools import partial
from pathlib import Path

from PIL import Image
import torch
from torchvision.transforms import v2 as T


def resize_bitmap(image, size):
    if image.size == (size, size):
        return image
    return image.resize((size, size), Image.Resampling.NEAREST)


def normalized_tensor():
    return T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True), T.Normalize((0.5,), (0.5,))])


def preprocess_description(size, labels, weight_hash):
    return {"version": 1, "format": "grayscale-nearest-square", "size": size,
            "class_order": list(labels), "model_sha256": weight_hash}


def checkpoint_preprocessing(stream, labels, legacy):
    """Return the transform, declared canvas and SHA-256 for this open state dictionary."""
    stream.seek(0)
    model_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    stream.seek(0)
    sidecar = Path(stream.name).with_suffix(".preprocess.json")
    if not sidecar.exists():
        return legacy, None, model_sha256
    info = json.loads(sidecar.read_text(encoding="utf-8"))
    size = info.get("size")
    if (info.get("version") != 1 or info.get("format") != "grayscale-nearest-square"
            or info.get("class_order") != list(labels) or type(size) is not int or not 1 <= size <= 2048):
        raise ValueError("The model preprocessing metadata is incompatible.")
    if model_sha256 != info.get("model_sha256"):
        raise ValueError("The preprocessing metadata belongs to a different model.")
    transform = T.Compose([partial(resize_bitmap, size=size), normalized_tensor()])
    return transform, size, model_sha256


def prediction_transform(weights, labels, legacy):
    weights = Path(weights)
    if not weights.with_suffix(".preprocess.json").exists():
        return legacy
    with weights.open("rb") as stream:
        transform, _, _ = checkpoint_preprocessing(stream, labels, legacy)
    return transform
