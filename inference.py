"""
AWS Lambda entry point for classifying one base64-encoded image.

Supported routes:
- GET / or /health: cheap container health check without loading PyTorch.
- GET/POST /warmup: load the model and run one tiny inference pass.
- POST /, /predict, or /classify: classify the supplied image.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
import sys
import time
from typing import Any

from shape_analyzer.inference_config import cfg

MODEL_PATH = cfg.MODEL_PATH.as_posix()
LABELS = list(cfg.LABELS)
IMG_SIZE = cfg.IMG_SIZE
MAX_REQ_BYTES = cfg.MAX_REQ_BYTES
MAX_IMAGE_PIXELS = 4096 * 4096
RESPONSE_HEADERS = {
    "Content-Type": "application/json",
    "Cache-Control": "no-store",
}
ROUTE_METHODS = {
    "/": ("GET", "POST"),
    "/health": ("GET",),
    "/warmup": ("GET", "POST"),
    "/predict": ("POST",),
    "/classify": ("POST",),
}

_model = None
_preprocess = None
_torch = None
_model_img_size = IMG_SIZE
_model_sha256 = None


def _json(status: int, body: dict[str, Any] | None = None, *, headers: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "statusCode": status,
        "headers": RESPONSE_HEADERS | (headers or {}),
        "body": json.dumps(body or {}, separators=(",", ":")),
        "isBase64Encoded": False,
    }


def _request_route(event: dict[str, Any]) -> tuple[str, str]:
    context = event.get("requestContext", {})
    if not isinstance(context, dict):
        raise ValueError("BadEvent")
    http = context.get("http", {})
    if not isinstance(http, dict):
        raise ValueError("BadEvent")
    default_method = "POST" if event.get("body") is not None else "GET"
    method = http.get("method", event.get("httpMethod", default_method))
    path = event.get("rawPath", event.get("path", http.get("path", "/")))
    if not isinstance(method, str) or not method.strip() or not isinstance(path, str) or not path.strip():
        raise ValueError("BadEvent")
    return method.strip().upper(), "/" + path.strip().strip("/")


def _health_payload() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "shape-analyzer",
        "model_loaded": _model is not None,
        "labels": LABELS,
        "img_size": _model_img_size if _model is not None else IMG_SIZE,
        "max_request_bytes": MAX_REQ_BYTES,
    }


def _load_model() -> None:
    global _model, _preprocess, _torch, _model_img_size, _model_sha256

    if _model is not None:
        return

    t0 = time.perf_counter()
    import torch
    from torchvision import transforms

    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "2")))

    from shape_analyzer.model import ShapeNet
    from shape_analyzer.preprocessing import checkpoint_preprocessing

    legacy = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    # Hash and deserialize the same open file if another process replaces the checkpoint.
    with open(MODEL_PATH, "rb") as stream:
        preprocess, canvas, digest = checkpoint_preprocessing(stream, LABELS, legacy)
        model = ShapeNet(n_cls=len(LABELS))
        model.load_state_dict(torch.load(stream, map_location="cpu", weights_only=True))
    model.eval()
    _preprocess = preprocess
    _model_img_size = canvas or IMG_SIZE
    _model_sha256 = digest
    _torch = torch
    _model = model

    print(f"[init] Model ready in {time.perf_counter() - t0:.2f}s", file=sys.stderr)


def _shape_confidences(img) -> dict[str, float]:
    _load_model()
    x = _preprocess(img).unsqueeze(0)
    with _torch.inference_mode():
        probs = _torch.softmax(_model(x), dim=1)[0]
    return dict(zip(LABELS, probs.tolist(), strict=True))


def _rounded(scores: dict[str, float]) -> dict[str, float]:
    return {label: round(probability, 4) for label, probability in scores.items()}


def _warmup_response(start: float) -> dict[str, Any]:
    from PIL import Image

    img = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)
    shape_confidences = _shape_confidences(img)
    return _json(200, {
        "status": "ready",
        "service": "shape-analyzer",
        "warmed": True,
        "model_loaded": _model is not None,
        "shape_confidences": _rounded(shape_confidences),
        "duration_ms": int((time.perf_counter() - start) * 1000),
    })


def _extract_b64(event: dict[str, Any]) -> str:
    raw_body = event.get("body")
    if raw_body is None:
        raise ValueError("MissingBody")
    if event.get("isBase64Encoded"):
        if not isinstance(raw_body, str):
            raise ValueError("BadBase64")
        return raw_body
    if isinstance(raw_body, str):
        try:
            raw_body = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise ValueError("BadJson") from exc
    if not isinstance(raw_body, dict):
        raise ValueError("BadJson")
    data = raw_body.get("b64", "")
    if not isinstance(data, str):
        raise ValueError("BadBase64")
    return data


def _classify_response(event: dict[str, Any], start: float) -> dict[str, Any]:
    try:
        body = _extract_b64(event)
    except ValueError as exc:
        return _json(400, {"error": str(exc)})
    if not body:
        return _json(400, {"error": "MissingImage"})
    if len(body) > MAX_REQ_BYTES:
        return _json(413, {"error": "ImageTooLarge"})

    try:
        img_bytes = base64.b64decode(body, validate=True)
    except (binascii.Error, ValueError):
        return _json(400, {"error": "BadBase64"})

    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(img_bytes)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                return _json(413, {"error": "ImageTooLarge"})
            img = source.convert("L")
    except Image.DecompressionBombError:
        return _json(413, {"error": "ImageTooLarge"})
    except (UnidentifiedImageError, OSError):
        return _json(400, {"error": "BadImage"})

    shape_confidences = _shape_confidences(img)
    top_shape, top_confidence = max(shape_confidences.items(), key=lambda item: item[1])
    return _json(200, {
        "class": top_shape,
        "confidence": round(top_confidence, 4),
        "shape_confidences": _rounded(shape_confidences),
        "duration_ms": int((time.perf_counter() - start) * 1000),
    })


def handler(event: dict[str, Any], _context) -> dict[str, Any]:
    start = time.perf_counter()
    if not isinstance(event, dict):
        return _json(400, {"error": "BadEvent"})

    try:
        method, path = _request_route(event)
    except ValueError:
        return _json(400, {"error": "BadEvent"})

    try:
        if path not in ROUTE_METHODS:
            return _json(404, {"error": "NotFound"})
        if method == "OPTIONS":
            return {
                "statusCode": 204,
                "headers": RESPONSE_HEADERS,
                "body": "",
                "isBase64Encoded": False,
            }
        if method not in ROUTE_METHODS[path]:
            allow = ", ".join((*ROUTE_METHODS[path], "OPTIONS"))
            return _json(405, {"error": "MethodNotAllowed"}, headers={"Allow": allow})
        if path == "/warmup":
            return _warmup_response(start)
        if method == "GET":
            return _json(200, _health_payload())
        return _classify_response(event, start)
    except Exception as exc:
        print("ERROR:", exc, file=sys.stderr)
        return _json(500, {"error": "ServerError"})
