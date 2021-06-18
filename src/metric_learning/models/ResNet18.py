import torch
from torch import nn
import torchvision as vision


# embeding_size = 512
class ResNet18(nn.Module):
    def __init__(self, path='', is_local=False):
        super().__init__()
        self.path = path
        self.is_local = is_local
        if self.is_local:
            self.backbone = torch.nn.Sequential(
                *(list(vision.models.resnet18(pretrained=False).children())[:-1]))
            self.load_state_dict(torch.load(self.path)['model_state_dict'])
        else:
            self.backbone = torch.nn.Sequential(
                *(list(vision.models.resnet18(pretrained=True).children())[:-1]))

    def forward(self, X):
        tmp = self.backbone(X)
        return tmp.view(tmp.size()[:2])
