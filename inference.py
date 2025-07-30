"""
AWS Lambda entry‑point for **image‑only** classification.

Request  (HTTP POST, JSON):
{
  "b64": "<base64‑encoded JPEG/PNG>"
}

Response (200):
{
  "class": "square",
  "confidence": 0.9921          # range 0‑1
}
"""

import base64, io, json, os, torch, glob
from PIL import Image
from torchvision import transforms

# ───────────────────────────── 1. Model & labels ──────────────────────────────
# Allow Lambda environment variable to override checkpoint path. If not set,
# automatically pick the first *.ckpt file from the `checkpoints` directory.
_model_path = os.getenv("MODEL_PATH")
if _model_path:
    MODEL_PATH = _model_path
else:
    ckpts = glob.glob(os.path.join("checkpoints", "*.ckpt"))
    MODEL_PATH = ckpts[0] if ckpts else "model.pt"
DEVICE     = "cpu"

model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval()

# Change to your model’s expected size / stats
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Class names in index order ─ replace with your own
LABELS = [
    "circle", "triangle", "square", "hexagon", "octagon",
]

# ───────────────────────────── 2. Lambda handler ──────────────────────────────
def handler(event, context):
    try:
        # ─── 1. Parse payload ───────────────────────────────────────────────
        if event.get("isBase64Encoded"):          # raw binary via Function URL
            img_bytes = base64.b64decode(event["body"])
        else:
            body = json.loads(event["body"])
            img_bytes = base64.b64decode(body["b64"])

        # ─── 2. Pre‑process & predict ──────────────────────────────────────
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        x   = preprocess(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
            conf, idx = torch.max(probs, 0)

        # ─── 3. Return result ──────────────────────────────────────────────
        return _resp(200, {
            "class": LABELS[idx],
            "confidence": round(conf.item(), 4),
        })

    except Exception as exc:
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
