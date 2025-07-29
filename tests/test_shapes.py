import numpy as np
from PIL import Image
from pathlib import Path
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from shape_analyzer.config import cfg
from shape_analyzer.helpers import _unit_vertices, generate_batch
from shape_analyzer.data import NPShapes


def test_unit_vertices_square():
    verts = _unit_vertices(4)
    assert verts.shape == (4, 2)


def test_generate_batch_circle(tmp_path):
    arr = generate_batch("circle", np.array([0, 50, 100], dtype=np.float32))
    assert arr.shape == (3, cfg.OUT_RES, cfg.OUT_RES)
    assert arr.dtype == np.uint8


def test_npshapes_dataset(tmp_path):
    imgs = np.zeros((2, cfg.OUT_RES, cfg.OUT_RES), dtype=np.uint8)
    cls = np.array([0, 1], dtype=np.int8)
    reg = np.array([0.1, 0.2], dtype=np.float32)
    npz_path = tmp_path / "test.npz"
    np.savez_compressed(npz_path, imgs=imgs, cls=cls, reg=reg)

    ds = NPShapes(npz=npz_path)
    assert len(ds) == 2
    img, y_cls, y_reg = ds[0]
    assert isinstance(img, Image.Image)
    assert int(y_cls) == 0
    from pytest import approx
    assert float(y_reg) == approx(0.1, rel=1e-5)
