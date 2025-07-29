"""
AWS Lambda entry‑point for **image‑only** classification.

Request  (HTTP POST, JSON):
{
  "b64": "<base64‑encoded JPEG/PNG>"
}

Response (200):
{
  "class": "cat",
  "confidence": 0.9921          # range 0‑1
}
"""

import base64, io, json, os, torch
from PIL import Image
from torchvision import transforms

# ───────────────────────────── 1. Model & labels ──────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "model.pt")   # override via Lambda env
DEVICE     = "cpu"

model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval()

# Change to your model’s expected size / stats
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# Class names in index order ─ replace with your own
LABELS = [
    "circle", "triangle", "square", "hexagon", "octagon",
]

# ───────────────────────────── 2. Lambda handler ──────────────────────────────
def handler(event, context):
    try:
        body = json.loads(event["body"])
        img_b64 = body["b64"]

        # Decode image → tensor
        img_bytes = base64.b64decode(img_b64)
        img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        x         = preprocess(img).unsqueeze(0).to(DEVICE)  # (1, C, H, W)

        # Forward pass
        with torch.no_grad():
            logits = model(x)
            probs  = torch.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, 0)

        return _resp(200, {
            "class": LABELS[idx],
            "confidence": round(conf.item(), 4),
        })

    except Exception as exc:
        # Log stack trace in CloudWatch; return generic error to caller
        print("ERROR:", exc)
        return _resp(500, {"error": str(exc)})

# ───────────────────────────── 3. Helper ──────────────────────────────────────
def _resp(status_code: int, body: dict):
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
        "isBase64Encoded": False,
    }
