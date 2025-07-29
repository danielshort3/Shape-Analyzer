from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torchvision import transforms as T

from .config import cfg
from .model import ShapeNet, ShapeLit


def interactive_demo(ckpt_path: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lit = ShapeLit.load_from_checkpoint(ckpt_path, map_location=device, model=ShapeNet())
    lit.eval().to(device)
    val_tfm = T.Compose([T.ToTensor(), T.Normalize([0.5], [0.5])])

    CANVAS, TH = cfg.OUT_RES, cfg.STROKE_PX
    canvas = np.zeros((CANVAS, CANVAS), np.uint8)
    drawing = False

    def _mouse(e, x, y, f, p):
        nonlocal drawing
        if e == cv2.EVENT_LBUTTONDOWN:
            drawing = True
        elif e == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.line(canvas, p[:2], (x, y), 255, TH)
            p[:2] = x, y
        elif e == cv2.EVENT_LBUTTONUP:
            drawing = False

    win = "Draw (s=pred r=reset q=quit)"
    cv2.namedWindow(win)
    pt = [-1, -1]
    cv2.setMouseCallback(win, _mouse, pt)
    while True:
        cv2.imshow(win, canvas)
        k = cv2.waitKey(1) & 255
        if k == ord("s") and canvas.max():
            small = cv2.resize(canvas, (cfg.DRAW_RES, cfg.DRAW_RES), cv2.INTER_AREA)
            inp = cv2.resize(small, (CANVAS, CANVAS), cv2.INTER_NEAREST)
            x = val_tfm(Image.fromarray(inp)).unsqueeze(0).to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                logits = lit(x)
            probs = logits.softmax(-1).cpu().numpy()[0]
            cls = probs.argmax()
            name = cfg.CLASSES[cls]
            plt.figure()
            plt.imshow(inp, cmap="gray")
            plt.axis("off")
            plt.text(
                2,
                16,
                f"{name} {probs[cls]*100:.1f}%",
                color="white",
                weight="bold",
                bbox=dict(fc="black", alpha=0.6, pad=3),
            )
            plt.show()
        elif k == ord("r"):
            canvas[:] = 0
        elif k == ord("q"):
            break
    cv2.destroyAllWindows()
