import torch
from torch import nn
import torchvision as vision


# embeding_size = 576
class MobilenetV3Small(nn.Module):
    def __init__(self, path='', is_local=False):
        super().__init__()
        self.path = path
        self.is_local = is_local
        model = vision.models.mobilenet_v3_small(pretrained=True)
        self.backbone = torch.nn.Sequential(
            *(list(model.children())[:-1]))

        if self.is_local:
            self.load_state_dict(torch.load(self.path)['model_state_dict'])

    def forward(self, X):
        tmp = self.backbone(X)
        return tmp.view(tmp.size()[:2])
