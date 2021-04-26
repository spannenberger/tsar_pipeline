import torch
from torch import nn


def replace_classifier(model, num_classes):
    classifier_name, old_classifier = model._modules.popitem()
    if isinstance(old_classifier, nn.Sequential):
        input_shape = old_classifier[-1].in_features
        old_classifier[-1] = nn.Linear(input_shape, num_classes)

    elif isinstance(old_classifier, nn.Linear):
        input_shape = old_classifier.in_features
        old_classifier = nn.Linear(input_shape, num_classes)
    else:
        raise Exception("Uknown type of classifier {}".format(type(old_classifier)))
    model.add_module(classifier_name, old_classifier)

class EfficientNetb0(nn.Module):
    def __init__(self, num_classes=10, path='', is_local=False):
        super().__init__()
        self.path = path
        self.is_local = is_local
        if self.is_local:
            self.backbone = torch.hub.load('narumiruna/efficientnet-pytorch', 'efficientnet_b0', pretrained=True)
            replace_classifier(self.backbone, num_classes)
            self.load_state_dict(torch.load(self.path)['model_state_dict'])
        else:
            self.backbone = torch.hub.load('narumiruna/efficientnet-pytorch', 'efficientnet_b0', pretrained=True)
            replace_classifier(self.backbone, num_classes)

    def forward(self, X):
        return self.backbone(X)


# test model
if __name__ == "__main__":
    model = EfficientNetb0()
    model.eval()
    input_tensor = torch.randn(1, 3, 256, 256)
    output = model(input_tensor)
    print(output)

