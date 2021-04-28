import torch
from torch import nn
from models.ClassifierReplacer import replace_classifier

class EfficientNetb3(nn.Module):
    def __init__(self, num_classes=10, path='', is_local=False, diff_classes_flag=False, old_num_classes=10):
        super().__init__()
        self.path = path
        self.diff_classes_flag = diff_classes_flag
        self.is_local = is_local
        if self.is_local:
            if self.diff_classes_flag:
                self.backbone = torch.hub.load('narumiruna/efficientnet-pytorch', 'efficientnet_b3', pretrained=False)
                replace_classifier(self.backbone, old_num_classes)
                self.load_state_dict(torch.load(self.path)['model_state_dict'])
                replace_classifier(self.backbone, num_classes)
            else:
                self.backbone = torch.hub.load('narumiruna/efficientnet-pytorch', 'efficientnet_b3', pretrained=False)
                replace_classifier(self.backbone, num_classes)
                self.load_state_dict(torch.load(self.path)['model_state_dict'])
        else:
            self.backbone = torch.hub.load('narumiruna/efficientnet-pytorch', 'efficientnet_b3', pretrained=True)
            replace_classifier(self.backbone, num_classes)

    def forward(self, X):
        return self.backbone(X)


# test model
if __name__ == "__main__":
    model = EfficientNetb3()
    model.eval()
    input_tensor = torch.randn(1, 3, 256, 256)
    output = model(input_tensor)
    print(output)

