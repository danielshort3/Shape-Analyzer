import torch
import lightning.pytorch as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from .config import cfg
from .data import build_qd_dataset, build_qd_holdout, make_loaders
from .model import ShapeNet, ShapeLit


def train():
    build_qd_dataset()
    build_qd_holdout()
    dl_tr, dl_va, _ = make_loaders()

    model = ShapeNet()
    lit = ShapeLit(model)

    early = EarlyStopping(monitor="val_loss", patience=3, mode="min")
    ckpt = ModelCheckpoint(
        dirpath=cfg.CKPT_DIR,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
    )

    trainer = L.Trainer(
        max_epochs=cfg.MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        precision=cfg.PRECISION,
        callbacks=[early, ckpt],
        log_every_n_steps=10,
    )
    trainer.fit(lit, dl_tr, dl_va)


if __name__ == "__main__":
    train()
