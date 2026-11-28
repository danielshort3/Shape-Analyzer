from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as Tv2
from .splits import prepare_splits
from .preprocessing import normalized_tensor, resize_bitmap
from tqdm.auto import tqdm

from .config import cfg, rng
from .helpers import generate_batch, load_npz


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
    print(f"âœ… Wrote {cfg.DATA_PATH}  ({imgs.shape[0]:,} samples)")


def build_qd_dataset(force: bool = False):
    """Return the disjoint development archive for the current source and configuration."""
    return prepare_splits(force=force).development


def build_qd_holdout(force: bool = False):
    """Return the paired holdout archive for the current source and configuration."""
    return prepare_splits(force=force).holdout


class NPShapes(Dataset):
    def __init__(self, npz=None, indices=None, tfm=None, *, arrays=None, resize_to=None):
        if arrays is None:
            if npz is None:
                npz = cfg.DATA_PATH if cfg.DATA_PATH.exists() else build_qd_dataset()
            arrays = load_npz(Path(npz))
        self.imgs, self.cls, self.reg = arrays["imgs"], arrays["cls"], arrays["reg"]
        self.source_ids = arrays.get("source_id")
        self.idxs = np.arange(len(self.imgs)) if indices is None else indices
        self.tfm = tfm
        self.resize_to = resize_to

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, i):
        j = self.idxs[i]
        img = Image.fromarray(self.imgs[j])
        if self.resize_to is not None:
            img = resize_bitmap(img, self.resize_to)
        if self.tfm:
            img = self.tfm(img)
        return img, torch.tensor(int(self.cls[j])), torch.tensor(float(self.reg[j]))


def add_pixel_noise(x):
    return x + (torch.rand_like(x) < 0.02).to(x.dtype) * (2 * torch.rand_like(x) - 1)


train_tfm = Tv2.Compose([
    Tv2.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.8, 1.2), fill=0),
    Tv2.GaussianBlur(3, (0.2, 1.2)),
    Tv2.ColorJitter(0.4, 0.4),
    Tv2.ToImage(),
    Tv2.ToDtype(torch.float32, scale=True),
    Tv2.RandomErasing(p=0.3, scale=(0.002, 0.01), ratio=(0.2, 5.0), value=0),
    add_pixel_noise,
    Tv2.Normalize((0.5,), (0.5,)),
])
val_tfm = normalized_tensor()


def make_loaders(files=None):
    files = files or prepare_splits()
    dev, test = load_npz(files.development), load_npz(files.holdout)
    if dev["signature"].item() != test["signature"].item():
        raise ValueError("Development and holdout archives come from different splits.")
    if not np.all(np.isin(dev["partition"], [0, 1])) or not np.all(test["partition"] == 2):
        raise ValueError("Unexpected training, validation or holdout partition codes.")
    for cid in range(len(cfg.CLASSES)):
        ids = np.concatenate([a["source_id"][a["cls"] == cid] for a in (dev, test)])
        if len(ids) != len(np.unique(ids)):
            raise ValueError("A source row appears more than once across the model partitions.")
    train = NPShapes(arrays=dev, indices=np.flatnonzero(dev["partition"] == 0),
                     tfm=train_tfm, resize_to=cfg.OUT_RES)
    valid = NPShapes(arrays=dev, indices=np.flatnonzero(dev["partition"] == 1),
                     tfm=val_tfm, resize_to=cfg.OUT_RES)
    holdout = NPShapes(arrays=test, tfm=val_tfm, resize_to=cfg.OUT_RES)

    def loader(dataset, shuffle):
        return DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=shuffle,
                          num_workers=cfg.WORKERS, pin_memory=torch.cuda.is_available(),
                          persistent_workers=cfg.WORKERS > 0)
    return loader(train, True), loader(valid, False), loader(holdout, False)
