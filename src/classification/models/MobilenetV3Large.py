import torch
from torch import nn
import torchvision as vision


class MobilenetV3Large(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = vision.models.mobilenet_v3_large(pretrained=True)
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, X):
        return self.backbone(X)


# test model
if __name__ == "__main__":
    model = MobilenetV3Large()
    model.eval()
    input_tensor = torch.randn(1, 3, 256, 256)
    output = model(input_tensor)

    print(output)
