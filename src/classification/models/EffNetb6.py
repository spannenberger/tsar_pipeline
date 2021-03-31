import torch
from torch import nn


class EffNetb6(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = torch.hub.load("rwightman/gen-efficientnet-pytorch", 'tf_efficientnet_b6_ns', pretrained=True)
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
        self.act = nn.Softmax(dim=1)

    def forward(self, X):
        return self.act(self.backbone(X))


# test model
if __name__ == "__main__":
    model = EffNetb6()
    input_tensor = torch.randn(1, 3, 256, 256)
    output = model(input_tensor)
    print(output)
