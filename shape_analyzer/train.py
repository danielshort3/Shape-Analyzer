from pathlib import Path
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

# Enable TensorFloat‑32 computation on supported hardware
torch.set_float32_matmul_precision("high")

from .config import cfg
from .data import build_qd_dataset, build_qd_holdout, make_loaders
from .model import ShapeNet


class LitShape(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = ShapeNet()
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, "min", patience=2, factor=0.5
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"},
        }


def train() -> Path:
    build_qd_dataset()
    build_qd_holdout()
    dl_tr, dl_va, _ = make_loaders()

    lit = LitShape()
    ckpt_cb = ModelCheckpoint(
        dirpath=cfg.CKPT_DIR,
        filename="model",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=cfg.MAX_EPOCHS,
        precision=cfg.PRECISION,
        callbacks=[ckpt_cb],
        default_root_dir=cfg.CKPT_DIR,
    )
    trainer.fit(lit, train_dataloaders=dl_tr, val_dataloaders=dl_va)

    ckpt_path = ckpt_cb.best_model_path
    state = torch.load(ckpt_path, map_location="cpu")
    weights = {k.replace("model.", ""): v for k, v in state["state_dict"].items()}
    out_path = cfg.CKPT_DIR / "model.pt"
    torch.save(weights, out_path)
    return out_path


if __name__ == "__main__":
    train()
