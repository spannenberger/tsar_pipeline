from torch import nn
import torchvision as vision
import torch


class ClassificationModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model

    def forward(self, X):
        return self.backbone(X)
