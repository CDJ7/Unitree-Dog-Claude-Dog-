"""Q1 models: ResNet18 transfer learning and a small CNN trained from scratch (baseline)."""
from __future__ import annotations

import torch
import torch.nn as nn


def resnet18_transfer(n_classes, pretrained=True):
    """ImageNet ResNet18 with the 1000-way fc replaced by an n_classes head (all layers fine-tuned)."""
    import torchvision
    if pretrained:
        from q1_supervised.pretrained import load_imagenet_resnet18
        m, source = load_imagenet_resnet18()
    else:
        m, source = torchvision.models.resnet18(weights=None), "random init"
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    m.weights_source = source
    return m


class SmallCNN(nn.Module):
    """4 conv blocks (32-64-128-256 ch, BN, ReLU, max-pool) + global average pool + linear; ~0.4 M params."""

    def __init__(self, n_classes):
        super().__init__()
        layers, c = [], 3
        for out in (32, 64, 128, 256):
            layers += [nn.Conv2d(c, out, 3, padding=1, bias=False), nn.BatchNorm2d(out), nn.ReLU(inplace=True),
                       nn.Conv2d(out, out, 3, padding=1, bias=False), nn.BatchNorm2d(out), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c = out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(c, n_classes))
        self.weights_source = "random init (trained from scratch)"

    def forward(self, x):
        return self.head(self.features(x))


def build_model(name, n_classes):
    if name == "resnet18":
        return resnet18_transfer(n_classes, pretrained=True)
    if name == "resnet18_scratch":
        return resnet18_transfer(n_classes, pretrained=False)
    if name == "smallcnn":
        return SmallCNN(n_classes)
    raise ValueError(name)


def count_params(m):
    return sum(p.numel() for p in m.parameters())
