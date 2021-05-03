import torch
from torch import nn
from catalyst.contrib import models


class ResNet50(nn.Module):
    def __init__(self, out_features=16):
        super().__init__()
        self.backbone = models.MnistSimpleNet(out_features=out_features)

    def forward(self, X):
        return self.backbone(X)
