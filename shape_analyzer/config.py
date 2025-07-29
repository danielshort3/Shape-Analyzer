from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

@dataclass
class CFG:
    # ----- data ---------------------------------------------------------
    CLASSES: tuple[str, ...] = (
        "circle",
        "triangle",
        "square",
        "hexagon",
        "octagon",
    )
    SIDE_MAP: dict[str, int] = field(
        default_factory=lambda: {
            "circle": 0,
            "triangle": 3,
            "square": 4,
            "hexagon": 6,
            "octagon": 8,
        }
    )
    OUT_RES: int = 256
    DRAW_RES: int = 64
    STROKE_PX: int = 10
    TRAIN_FRAC: float = 0.9
    N_BINS: int = 10
    N_PER_SHAPE: int = 50_000
    HOLD_QD_PER_SHAPE: int = 10_000
    RNG_SEED: int = 42

    # optimiser / schedule ----------------------------------------------
    MAX_EPOCHS: int = 100
    BATCH_SIZE: int = 1024
    LR: float = 3e-4
    WD: float = 1e-4
    PRECISION: str = "16-mixed"

    # file paths ---------------------------------------------------------
    DATA_PATH: Path = Path("shapes_data.npz")
    HOLD_PATH: Path = Path("qd_holdout.npz")
    CACHE_DIR: Path = Path("qd_cache")
    CKPT_DIR: Path = Path("checkpoints")


cfg = CFG()

rng = np.random.default_rng(cfg.RNG_SEED)
