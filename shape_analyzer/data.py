from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as Tv2
from tqdm.auto import tqdm

from .config import cfg, rng
from .helpers import generate_batch, load_npz, _qd_array


def build_synthetic_dataset(force: bool = False):
    if cfg.DATA_PATH.exists() and not force:
        return
    imgs, cls, reg = [], [], []
    for cid, name in tqdm(enumerate(cfg.CLASSES), total=len(cfg.CLASSES)):
        for b in tqdm(range(cfg.N_BINS), total=cfg.N_BINS, leave=False):
            tgt = rng.uniform(b * 10, (b + 1) * 10, cfg.N_PER_SHAPE // cfg.N_BINS)
            imgs.append(generate_batch(name, tgt))
            cls.append(np.full_like(tgt, cid, np.int8))
            reg.append(tgt / 100.0)
    imgs = np.concatenate(imgs, axis=0)
    cls = np.concatenate(cls, dtype=np.int8)
    reg = np.concatenate(reg).astype(np.float32)
    np.savez_compressed(cfg.DATA_PATH, imgs=imgs, cls=cls, reg=reg)
    print(f"✅ Wrote {cfg.DATA_PATH}  ({imgs.shape[0]:,} samples)")


def build_qd_dataset(force: bool = False):
    """Build dataset from Google QuickDraw sketches."""
    if cfg.DATA_PATH.exists() and not force:
        return
    imgs, cls, reg = [], [], []
    print("Building QuickDraw dataset …")
    for shape in tqdm(cfg.CLASSES, total=len(cfg.CLASSES)):
        src = _qd_array(shape)
        choice = rng.choice(len(src), cfg.N_PER_SHAPE, replace=False)
        for idx in choice:
            arr = src[idx].reshape(28, 28)
            img = Image.fromarray(arr).resize((cfg.OUT_RES, cfg.OUT_RES), Image.NEAREST)
            imgs.append(np.asarray(img, np.uint8))
            cls.append(cfg.CLASSES.index(shape))
            reg.append(-1.0)
    imgs = np.stack(imgs, axis=0).astype(np.uint8)
    cls = np.asarray(cls, np.int8)
    reg = np.asarray(reg, np.float32)
    np.savez_compressed(cfg.DATA_PATH, imgs=imgs, cls=cls, reg=reg)
    print(f"✅ Wrote {cfg.DATA_PATH}  ({len(imgs):,} samples)")


def build_qd_holdout(force: bool = False):
    if cfg.HOLD_PATH.exists() and not force:
        return
    imgs, cls, reg = [], [], []
    print("Building QuickDraw hold‑out …")
    for shape in tqdm(cfg.CLASSES, total=len(cfg.CLASSES)):
        src = _qd_array(shape)
        choice = rng.choice(len(src), cfg.HOLD_QD_PER_SHAPE, replace=False)
        for idx in choice:
            arr = src[idx].reshape(28, 28)
            img = Image.fromarray(arr).resize((cfg.OUT_RES, cfg.OUT_RES), Image.NEAREST)
            imgs.append(np.asarray(img, np.uint8))
            cls.append(cfg.CLASSES.index(shape))
            reg.append(-1.0)
    imgs = np.stack(imgs, axis=0).astype(np.uint8)
    cls = np.asarray(cls, np.int8)
    reg = np.asarray(reg, np.float32)
    np.savez_compressed(cfg.HOLD_PATH, imgs=imgs, cls=cls, reg=reg)
    print(f"✅ Wrote {cfg.HOLD_PATH}  ({len(imgs):,} samples)")


class NPShapes(Dataset):
    def __init__(self, npz: Path = cfg.DATA_PATH, indices=None, tfm=None):
        z = load_npz(npz)
        self.imgs, self.cls, self.reg = z["imgs"], z["cls"], z["reg"]
        self.idxs = np.arange(len(self.imgs)) if indices is None else indices
        self.tfm = tfm

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        j = self.idxs[i]
        img = Image.fromarray(self.imgs[j])
        if self.tfm:
            img = self.tfm(img)
        import torch
        return img, torch.tensor(int(self.cls[j])), torch.tensor(float(self.reg[j]))


train_tfm = Tv2.Compose([
    Tv2.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.8, 1.2), fill=0),
    Tv2.GaussianBlur(3, (0.2, 1.2)),
    Tv2.RandomErasing(p=0.3, scale=(0.002, 0.01), ratio=(0.2, 5.0), value=0),
    Tv2.ColorJitter(0.4, 0.4),
    Tv2.ToImage(),
    Tv2.ToDtype(torch.float32, scale=True),
    Tv2.Lambda(lambda x: x + (np.random.rand(*x.shape) < 0.02).astype(x.dtype) * (2 * np.random.rand(*x.shape) - 1)),
    Tv2.Normalize((0.5,), (0.5,)),
])

val_tfm = Tv2.Compose([
    Tv2.ToImage(),
    Tv2.ToDtype(torch.float32, scale=True),
    Tv2.Normalize((0.5,), (0.5,)),
])


def make_loaders():
    all_npz = load_npz(cfg.DATA_PATH)
    N = len(all_npz["cls"])
    perm = rng.permutation(N)
    split = int(N * cfg.TRAIN_FRAC)
    tr_idx, va_idx = perm[:split], perm[split:]
    ds_tr = NPShapes(indices=tr_idx, tfm=train_tfm)
    ds_va = NPShapes(indices=va_idx, tfm=val_tfm)
    ds_ho = NPShapes(npz=cfg.HOLD_PATH, tfm=val_tfm)

    def loader(ds, shuf):
        return DataLoader(
            ds,
            batch_size=cfg.BATCH_SIZE,
            shuffle=shuf,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )

    return loader(ds_tr, True), loader(ds_va, False), loader(ds_ho, False)
