import torch
import torch.nn as nn
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


class ShapeLit(nn.Module):
    """Compatibility wrapper used by older demos."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)
