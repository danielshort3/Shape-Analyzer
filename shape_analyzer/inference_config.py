from dataclasses import dataclass
from pathlib import Path
import os

@dataclass(frozen=True)
class InferenceCFG:
    MODEL_PATH: Path = Path(os.getenv("MODEL_PATH", "/opt/python/model.pt"))
    LABELS: tuple[str, ...] = (
        "circle",
        "triangle",
        "square",
        "hexagon",
        "octagon",
    )
    IMG_SIZE: int = 224
    MAX_REQ_BYTES: int = 500_000


cfg = InferenceCFG()
