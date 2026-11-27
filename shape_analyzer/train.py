import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# Enable TensorFloat‑32 computation on supported hardware
torch.set_float32_matmul_precision("high")

from .config import cfg
from .data import make_loaders
from .splits import prepare_splits
from .evaluation import evaluate
from .preprocessing import preprocess_description
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
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            "min",
            patience=cfg.LR_PATIENCE,
            factor=0.5,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"},
        }


def _write_json(path, value):
    with NamedTemporaryFile(dir=path.parent, mode="w", delete=False, encoding="utf-8") as handle:
        temp = Path(handle.name)
    try:
        temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def train(*, sources=None, source_name="Google Quick, Draw!", accelerator="auto", progress=True) -> Path:
    """Select a checkpoint on validation loss, evaluate it once on the held-out split, and export."""
    L.seed_everything(cfg.RNG_SEED, workers=True)
    files = prepare_splits(sources, source_name=source_name)
    train_loader, valid_loader, test_loader = make_loaders(files)
    lit = LitShape()
    checkpoint = ModelCheckpoint(dirpath=cfg.CKPT_DIR, filename="model", save_top_k=1,
                                 monitor="val_loss", mode="min", save_weights_only=True)
    early = EarlyStopping(monitor="val_loss", patience=cfg.EARLY_STOP_PATIENCE, mode="min")
    use_cpu = accelerator == "cpu" or (accelerator == "auto" and not torch.cuda.is_available())
    precision = "32-true" if use_cpu else cfg.PRECISION
    trainer = L.Trainer(max_epochs=cfg.MAX_EPOCHS, precision=precision, accelerator=accelerator,
                        devices=1, callbacks=[checkpoint, early], logger=False,
                        enable_progress_bar=progress, default_root_dir=cfg.CKPT_DIR)
    trainer.fit(lit, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    state = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=True)
    lit.load_state_dict(state["state_dict"])
    metrics = evaluate(lit.model, train_loader.dataset, test_loader, cfg.CLASSES)
    metrics.update({"split": files.spec, "image_size": cfg.OUT_RES,
                    "best_validation_loss": float(checkpoint.best_model_score),
                    "training_epochs": int(trainer.current_epoch), "precision": precision,
                    "hardware": str(trainer.strategy.root_device), "torch": str(torch.__version__)})
    cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.CKPT_DIR / "model.pt"
    with NamedTemporaryFile(dir=cfg.CKPT_DIR, suffix=".pt", delete=False) as handle:
        temp = Path(handle.name)
    try:
        torch.save({k: v.detach().cpu() for k, v in lit.model.state_dict().items()}, temp)
        with temp.open("rb") as handle:
            metrics["model_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
        contract = preprocess_description(cfg.OUT_RES, cfg.CLASSES, metrics["model_sha256"])
        _write_json(out.with_suffix(".preprocess.json"), contract)
        os.replace(temp, out)
    finally:
        temp.unlink(missing_ok=True)
    _write_json(cfg.CKPT_DIR / "evaluation.json", metrics)
    return out


if __name__ == "__main__":
    train()
