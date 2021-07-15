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


class MobilenetV3Small(MetricLearningModel):
    def __init__(self, path='', is_local=False):
        pretrained = False if is_local else True
        model = vision.models.mobilenet_v3_small(pretrained=pretrained)
        super().__init__(model, path, is_local)

class ResNet18(MetricLearningModel):
    def __init__(self, path='', is_local=False):
        pretrained = False if is_local else True
        model = vision.models.resnet18(pretrained=pretrained)
        super().__init__(model, path, is_local)

class ResNext50(MetricLearningModel):
    def __init__(self, path='', is_local=False):
        pretrained = False if is_local else True
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=pretrained)
        super().__init__(model, path, is_local)

if __name__ == '__main__':
    model = ResNet18()
    model.eval()