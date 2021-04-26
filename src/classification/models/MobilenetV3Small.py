import torch
from torch import nn
import torchvision as vision

def new_classifier(model, num_classes):
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

class MobilenetV3Small(nn.Module):
    def __init__(self, num_classes=10, path='', is_local=False):
        super().__init__()
        self.path = path
        self.is_local = is_local
        if self.is_local:
            print('Start loading local model...')
            self.backbone = vision.models.mobilenet_v3_small(pretrained=False)
            new_classifier(self.backbone, num_classes)
            self.load_state_dict(torch.load(self.path)['model_state_dict'])
        else:
            self.backbone = vision.models.mobilenet_v3_small(pretrained=True)
            new_classifier(self.backbone, num_classes)
    def forward(self, X):
        return self.backbone(X)


# test model
if __name__ == "__main__":
    model = MobilenetV3Small()
    model.eval()
    input_tensor = torch.randn(1, 3, 256, 256)
    output = model(input_tensor)

    print(output)
