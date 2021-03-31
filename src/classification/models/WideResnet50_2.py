import torch
from torch import nn


class WideResnet50_2(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)

        # replace classifier
        classifier_name, old_classifier = self.backbone._modules.popitem()
        if isinstance(old_classifier, nn.Sequential):
            input_shape = old_classifier[-1].in_features
            old_classifier[-1] = nn.Linear(input_shape, num_classes)

        elif isinstance(old_classifier, nn.Linear):
            input_shape = old_classifier.in_features
            old_classifier = nn.Linear(input_shape, num_classes)
        else:
            raise Exception("Uknown type of classifier {}".format(type(old_classifier)))
        self.backbone.add_module(classifier_name, old_classifier)
        # end

    def forward(self, X):
        return self.backbone(X)


# test model
if __name__ == "__main__":
    model = WideResnet50_2()
    model.eval()
    input_tensor = torch.randn(1, 3, 256, 256)
    output = model(input_tensor)
    print(output)
