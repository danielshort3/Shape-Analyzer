import torch
import torch.nn as nn
import lightning.pytorch as L
import torchmetrics
from torchvision.models import resnet18

from .config import cfg


class ShapeNet(nn.Module):
    """Simple ResNet18 encoder with a single classification head."""

    def __init__(self, n_cls: int = len(cfg.CLASSES), dropout_p: float = 0.4):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder = nn.Sequential(*list(base.children())[:-2])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout_p)
        dim = base.fc.in_features
        self.cls_head = nn.Linear(dim, n_cls)

    def forward(self, x):
        f = self.drop(self.gap(self.encoder(x)).flatten(1))
        return self.cls_head(f)


class ShapeLit(L.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.loss_cls = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=len(cfg.CLASSES))
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=len(cfg.CLASSES))

    def forward(self, x):
        return self.model(x)

    def _step(self, batch, stage):
        x, yc, *yr = batch
        logits = self(x)
        cls_loss = self.loss_cls(logits, yc)
        loss = cls_loss
        acc = self.train_acc if stage == "train" else self.val_acc
        acc.update(logits.softmax(-1), yc)
        self.log_dict(
            {f"{stage}_{k}": v for k, v in dict(loss=loss, cls_loss=cls_loss, acc=acc).items()},
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    training_step = lambda self, b, _: self._step(b, "train")
    validation_step = lambda self, b, _: self._step(b, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", patience=2, factor=0.5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}
