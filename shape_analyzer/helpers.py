from __future__ import annotations
import functools
import numpy as np
from skimage.draw import polygon_perimeter
from skimage.transform import resize as sk_resize
from PIL import Image
from pathlib import Path
import requests

from .config import cfg, rng


@functools.lru_cache(maxsize=None)
def _unit_vertices(n_sides: int) -> np.ndarray:
    """Return vertices of a regular polygon or circle on the unit circle."""
    ang = np.linspace(0, 2 * np.pi, n_sides, endpoint=False, dtype=np.float32)
    return np.stack([np.cos(ang), np.sin(ang)], 1)


def _curve_amp(reg_pct: float, amp_min: float = 0.0, amp_max: float = 0.35, gamma: float = 0.60):
    frac = 1.0 - reg_pct / 100
    return amp_min + (amp_max - amp_min) * (frac ** gamma)


def _chaikin(pts: np.ndarray, n_iter: int = 1) -> np.ndarray:
    for _ in range(n_iter):
        Q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        R = 0.25 * pts[:-1] + 0.75 * pts[1:]
        pts = np.vstack([Q.T, R.T]).T.reshape(-1, 2)
    return pts


def _sample_vertices(n_sides: int, reg_pct: float, n_circle_pts: int = 128):
    amp = _curve_amp(reg_pct)
    if n_sides == 0:
        base = _unit_vertices(n_circle_pts)
        if reg_pct >= 50:
            return base
        k = 1 + (50 - reg_pct) / 20
        sx, sy = 1 + rng.normal(0, amp * 1.5 * k, 2)
        shear = rng.normal(0, amp * 0.8 * k)
        affine = np.array([[sx, shear], [0, sy]], np.float32)
        verts = (affine @ base.T).T
        n_lobes = rng.integers(1, 4)
        phase = rng.uniform(0, 2 * np.pi)
        ripple = 1 + amp * 0.6 * k * np.sin(
            np.linspace(phase, phase + 2 * np.pi * n_lobes, n_circle_pts, dtype=np.float32)
        )
        verts *= ripple[:, None]
        return _chaikin(verts, 1)

    base = _unit_vertices(n_sides)
    radial = rng.normal(1.0, amp, n_sides).astype(np.float32)
    angle = rng.normal(0.0, amp * 0.6, n_sides).astype(np.float32)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.stack([c, s, -s, c], -1).reshape(-1, 2, 2)
    verts = (rot @ base[:, :, None]).squeeze(-1) * radial[:, None]
    if reg_pct >= 50:
        return verts

    bulged = []
    bulge = amp * 0.8 * min(1.0, n_sides / 6)
    for i in range(n_sides):
        v0, v1 = verts[i], verts[(i + 1) % n_sides]
        mid = (v0 + v1) / 2
        nrm = np.array([v1[1] - v0[1], v0[0] - v1[0]], np.float32)
        nrm /= np.linalg.norm(nrm) + 1e-7
        bulged.extend([v0, mid + bulge * nrm])
    return _chaikin(np.array(bulged, np.float32), 1)


def _scale_from_reg(reg_pct: float, r_min: float = 0.20, r_max: float = 0.40) -> float:
    return r_min + (r_max - r_min) * (reg_pct / 100)


def _rasterise(verts: np.ndarray, canvas: np.ndarray, pct_scale: float):
    h, w = canvas.shape
    cx, cy = w // 2, h // 2
    r = h * pct_scale
    pts = verts * r + np.array([cx, cy], np.float32)
    rr, cc = polygon_perimeter(pts[:, 1], pts[:, 0], shape=canvas.shape, clip=True)
    canvas[rr, cc] = 255


def generate_batch(shape: str, reg_pct_array: np.ndarray) -> np.ndarray:
    n_sides = cfg.SIDE_MAP[shape]
    B = len(reg_pct_array)
    canv = np.zeros((B, cfg.DRAW_RES, cfg.DRAW_RES), np.uint8)
    for i, rpct in enumerate(reg_pct_array):
        verts = _sample_vertices(n_sides, rpct)
        _rasterise(verts, canv[i], _scale_from_reg(rpct))
    imgs = sk_resize(canv, (B, cfg.OUT_RES, cfg.OUT_RES), order=0, preserve_range=True).astype(np.uint8)
    return imgs


cfg.CACHE_DIR.mkdir(exist_ok=True)
_NPZ_CACHE: dict[str, np.lib.npyio.NpzFile] = {}


def load_npz(path: Path):
    if path.as_posix() not in _NPZ_CACHE:
        _NPZ_CACHE[path.as_posix()] = np.load(path, mmap_mode="r")
    return _NPZ_CACHE[path.as_posix()]


@functools.lru_cache(maxsize=None)
def _qd_array(shape: str):
    p = cfg.CACHE_DIR / f"{shape}.npy"
    if not p.exists():
        url = (
            "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
            f"{shape}.npy"
        )
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(p, "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
    return np.load(p, mmap_mode="r")
