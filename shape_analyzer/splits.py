"""Versioned Quick, Draw! splits with disjoint, reproducible per-class source rows."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile

import numpy as np

from .config import cfg
from .helpers import _qd_array


@dataclass(frozen=True)
class SplitFiles:
    development: Path
    holdout: Path
    spec: dict


def _write_archive(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            temp = Path(stream.name)
            np.savez_compressed(stream, **arrays)
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def _matches(path, signature, ids, classes, partitions):
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as saved:
            return (
                {"imgs", "reg", "spec"} <= set(saved.files)
                and saved["signature"].item() == signature
                and np.array_equal(saved["source_id"], ids)
                and np.array_equal(saved["cls"], classes)
                and np.array_equal(saved["partition"], partitions)
            )
    except (KeyError, OSError, ValueError, BadZipFile):
        return False


def prepare_splits(sources=None, *, source_name="Google Quick, Draw!", force=False):
    """Allocate a single seeded choice per class, then partition it without replacement.

    Legacy anonymous archives remain readable through NPShapes(path). Training always
    uses these versioned archives, whose identities cover the complete input arrays.
    """
    if sources is None:
        sources = {label: _qd_array(label) for label in cfg.CLASSES}
    dev_count, test_count = cfg.N_PER_SHAPE, cfg.HOLD_QD_PER_SHAPE
    train_count = int(dev_count * cfg.TRAIN_FRAC)
    if not 0 < train_count < dev_count or test_count < 1:
        raise ValueError("Each class needs at least one training, validation and test drawing.")
    specs = []
    selections = []
    for cid, label in enumerate(cfg.CLASSES):
        source = sources[label]
        if source.dtype != np.uint8 or source.ndim != 2 or source.shape[1] != 784:
            raise ValueError(f"{label} must be a uint8 28 x 28 bitmap matrix.")
        if len(source) < dev_count + test_count:
            raise ValueError(f"{label} needs {dev_count + test_count} distinct source rows.")
        digest = hashlib.sha256(memoryview(np.ascontiguousarray(source)).cast("B")).hexdigest()
        specs.append({"class": label, "rows": len(source), "bitmap_sha256": digest})
        rng = np.random.default_rng([cfg.RNG_SEED, cid])
        selections.append(rng.choice(len(source), dev_count + test_count, replace=False))
    spec = {"version": 2, "source": source_name, "seed": cfg.RNG_SEED,
            "train_per_class": train_count, "validation_per_class": dev_count - train_count,
            "test_per_class": test_count, "sources": specs}
    description = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    signature = hashlib.sha256(description.encode()).hexdigest()
    files = []
    for role, base, cut, per_class in (
        ("development", cfg.DATA_PATH, slice(0, dev_count), [0] * train_count + [1] * (dev_count - train_count)),
        ("holdout", cfg.HOLD_PATH, slice(dev_count, None), [2] * test_count),
    ):
        path = base.with_name(f"{base.stem}.{role}.v2-{signature[:20]}.npz")
        role_ids = [ids[cut] for ids in selections]
        ids = np.concatenate(role_ids)
        classes = np.repeat(np.arange(len(cfg.CLASSES), dtype=np.int16), len(per_class))
        partitions = np.tile(np.asarray(per_class, dtype=np.uint8), len(cfg.CLASSES))
        if force or not _matches(path, signature, ids, classes, partitions):
            samples = [sources[label][indices] for label, indices in zip(cfg.CLASSES, role_ids)]
            _write_archive(path, imgs=np.concatenate(samples).reshape(-1, 28, 28),
                           cls=classes, reg=np.full(len(ids), -1, dtype=np.float32),
                           source_id=ids, partition=partitions, signature=signature, spec=description)
        files.append(path)
    return SplitFiles(*files, spec)
