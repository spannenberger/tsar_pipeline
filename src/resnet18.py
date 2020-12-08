import torch
from torch import nn
import torchvision as vision


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = vision.models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, num_classes)

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, X):
        return self.backbone(X)


# test model
if __name__ == "__main__":
    model = ResNet18()
    input_tensor = torch.randn(1, 3, 256, 256)
    output = model(input_tensor)

    print(output)