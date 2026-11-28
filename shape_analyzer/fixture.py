"""Small independently drawn five-class fixture for an offline training smoke run."""
import math

import numpy as np
from PIL import Image, ImageDraw

from .config import cfg


def geometry_sources(count, seed):
    sources = {}
    for cid, label in enumerate(cfg.CLASSES):
        rng = np.random.default_rng([seed, cid, 701])
        images = []
        for _ in range(count):
            image = Image.new("L", (28, 28))
            draw = ImageDraw.Draw(image)
            x, y = rng.integers(12, 16, size=2)
            rx, ry = rng.integers(7, 12, size=2)
            width = int(rng.integers(1, 3))
            sides = cfg.SIDE_MAP[label]
            if sides == 0:
                draw.ellipse((x-rx, y-ry, x+rx, y+ry), outline=255, width=width)
            else:
                angle = rng.uniform(-0.35, 0.35)
                points = [(x + rx*math.cos(angle + i*math.tau/sides),
                           y + ry*math.sin(angle + i*math.tau/sides)) for i in range(sides)]
                draw.line(points + [points[0]], fill=255, width=width)
            images.append(np.asarray(image).reshape(-1))
        sources[label] = np.stack(images)
    return sources
