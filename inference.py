"""
Lambda entry‑point (`handler`) for image‑ or JSON‑based inference.

✔ Loads the model once at import‑time (cold‑start)  
✔ Accepts a JSON POST body from API Gateway  
   • For an **image** request:   {"type":"image","b64":"..."}  
   • For a **text**  request:   {"type":"text", "text":"Hello"}  
✔ Returns JSON: {"pred": ..., "prob": [...]} or {"generated": "..."}
"""

import base64, io, json, os, torch
from PIL import Image
from torchvision import transforms

# ----------------------------------------------------------------------
# 1. Load model & preprocessing ONCE (cold‑start cost only)
# ----------------------------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "model.pt")   # override in Lambda if desired
DEVICE     = "cpu"

model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
model.eval()

img_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# ----------------------------------------------------------------------
# 2. Lambda handler
# ----------------------------------------------------------------------

def handler(event, context):
    try:
        body = json.loads(event["body"])
        req_type = body.get("type", "text")

        if req_type == "image":
            # Expect base64‑encoded JPEG/PNG
            img_bytes = base64.b64decode(body["b64"])
            img       = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            x         = img_tfm(img).unsqueeze(0)          # (1,C,H,W)

        elif req_type == "text":
            # Simple numeric vectorisation example (replace with tokenizer)
            import numpy as np
            vec = np.array([ord(c) for c in body["text"]], dtype="float32")
            x   = torch.tensor(vec).unsqueeze(0)            # (1,seq)

        else:
            return _resp(400, {"error": f"unknown type '{req_type}'"})

        with torch.no_grad():
            y = model(x.to(DEVICE))

        # Example for classification logits
        if y.ndim == 2:
            probs = torch.softmax(y, dim=1)[0]
            pred  = int(torch.argmax(probs).item())
            return _resp(200, {"pred": pred, "prob": probs.tolist()})
        # Example for text generation tensor -> string
        elif y.ndim == 1:
            out_txt = "".join(map(chr, y.int().tolist()))
            return _resp(200, {"generated": out_txt})
        else:
            return _resp(500, {"error": "unexpected model output shape"})

    except Exception as exc:
        return _resp(500, {"error": str(exc)})

# ----------------------------------------------------------------------
# 3. Helper
# ----------------------------------------------------------------------

def _resp(status_code: int, body: dict):
    """Format response for Lambda + API Gateway HTTP API"""
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
        "isBase64Encoded": False,
    }
