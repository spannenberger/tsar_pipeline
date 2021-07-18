import torch
from torch import nn
import torchvision as vision

class MetricLearningModel(nn.Module):
    def __init__(self, model, path='', is_local=False):
        super().__init__()
        self.path = path
        self.is_local = is_local
        self.backbone = torch.nn.Sequential(
            *(list(model.children())[:-1]))
        if self.is_local:
            self.load_state_dict(torch.load(self.path)['model_state_dict'])

    @property
    def embedding_size(self):
        if not hasattr(self, "embeddings"):
            switch = False
            if self.backbone.training:
                switch = True
            self.backbone.eval()
            input_tensor = torch.randn(1, 3, 256, 256)
            self.embeddings = self.backbone(input_tensor).shape[1]
            if switch:
                self.backbone.train()
        return self.embeddings

    def forward(self, X):
        tmp = self.backbone(X)
        return tmp.view(tmp.size()[:2])
