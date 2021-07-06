import torch
from torch import nn


# embeding_size = 2048
class ResNext50(nn.Module):
    def __init__(self, path='', is_local=False):
        super().__init__()
        self.path = path
        self.is_local = is_local
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True)
        self.backbone = torch.nn.Sequential(
                *(list(model.children())[:-1]))
        if self.is_local:
            self.load_state_dict(torch.load(self.path)['model_state_dict'])

    def forward(self, X):
        tmp = self.backbone(X)
        return tmp.view(tmp.size()[:2])
