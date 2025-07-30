"""
λ‑Function entry‑point: classify one base‑64 image.

POST body:
{
  "b64": "<base64‑image>"
}

Returns 200:
{
  "class": "square",
  "confidence": 0.9921
}
"""

from __future__ import annotations
import base64, io, json, os, sys, time
from typing import Any
from shape_analyzer.inference_config import cfg

# ───────────────────────────── Constants ──────────────────────────────
MODEL_PATH      = cfg.MODEL_PATH.as_posix()
LABELS          = list(cfg.LABELS)
IMG_SIZE        = cfg.IMG_SIZE                   # final crop‑size for the model
MAX_REQ_BYTES   = cfg.MAX_REQ_BYTES              # reject > 500 kB payloads

# Globals – initialised lazily inside _load_model()
_model          = None                 # torch.nn.Module
_preprocess     = None                 # torchvision.Transform
_torch          = None                 # reference to torch module

# ───────────────────────────── Utilities ──────────────────────────────
def _json(status: int, body: dict[str, Any]) -> dict[str, Any]:
    """API Gateway proxy‑style response."""
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body, separators=(",", ":")),
        "isBase64Encoded": False,
    }


def _load_model():
    """Load Torch + weights exactly once per container."""
    global _model, _preprocess, _torch

    if _model is not None:
        return

    t0 = time.time()
    import torch                         # heavy import
    from torchvision import transforms   # heavy import

    _torch = torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "2")))

    from shape_analyzer.model import ShapeNet

    model = ShapeNet(n_cls=len(LABELS))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    _model = model
    _preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    print(f"[init] Model ready in {time.time()-t0:.2f}s", file=sys.stderr)


# ───────────────────────────── Lambda handler ─────────────────────────
def handler(event: dict[str, Any], _context) -> dict[str, Any]:
    try:
        _load_model()                              # ensure model is live

        # 1 Decode payload
        body = event["body"] if event.get("isBase64Encoded") else json.loads(event["body"])["b64"]
        if len(body) > MAX_REQ_BYTES:
            return _json(413, {"error": "ImageTooLarge"})

        try:
            img_bytes = base64.b64decode(body)
        except (base64.binascii.Error, TypeError):
            return _json(400, {"error": "BadBase64"})

        from PIL import Image, UnidentifiedImageError
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("L")
        except (UnidentifiedImageError, OSError):
            return _json(400, {"error": "BadImage"})

        # 2 Pre‑process & predict
        x = _preprocess(img).unsqueeze(0)          # (1, C, H, W)
        with _torch.no_grad():
            probs = _torch.softmax(_model(x), dim=1)[0]
            conf, idx = _torch.max(probs, 0)

        return _json(200, {
            "class": LABELS[int(idx)],
            "confidence": round(conf.item(), 4),
        })

    except Exception as exc:
        print("⛔ ERROR:", exc, file=sys.stderr)
        return _json(500, {"error": "ServerError"})
