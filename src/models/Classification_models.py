from torch import nn
import torchvision as vision
import torch

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

class ClassificationModel(nn.Module):
    def __init__(self, model, num_classes=10, path='', is_local=False, diff_classes_flag=False, old_num_classes=10):
        super().__init__()
        self.path = path
        self.diff_classes_flag = diff_classes_flag
        self.is_local = is_local
        self.backbone = model
        if self.is_local:
            if self.diff_classes_flag:
                replace_classifier(self.backbone, old_num_classes)
                self.load_state_dict(torch.load(self.path)['model_state_dict'])
                replace_classifier(self.backbone, num_classes)
            else:
                replace_classifier(self.backbone, num_classes)
                self.load_state_dict(torch.load(self.path)['model_state_dict'])
        else:
            replace_classifier(self.backbone, num_classes)
    def forward(self, X):
        return self.backbone(X)