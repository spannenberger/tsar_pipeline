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

class MobilenetV3Small(ClassificationModel):
    def __init__(self, num_classes=10, path='', is_local=False, diff_classes_flag=False, old_num_classes=10):
        pretrained = False if is_local else True
        model = vision.models.mobilenet_v3_small(pretrained=pretrained)
        super().__init__(model, num_classes, path, is_local, diff_classes_flag, old_num_classes)

class MobilenetV3Large(ClassificationModel):
    def __init__(self, num_classes=10, path='', is_local=False, diff_classes_flag=False, old_num_classes=10):
        pretrained = False if is_local else True
        model = vision.models.mobilenet_v3_large(pretrained=pretrained)
        super().__init__(model, num_classes, path, is_local, diff_classes_flag, old_num_classes)

class MobilenetV2(ClassificationModel):
    def __init__(self, num_classes=10, path='', is_local=False, diff_classes_flag=False, old_num_classes=10):
        pretrained = False if is_local else True
        model = vision.models.mobilenet_v2(pretrained=pretrained)
        super().__init__(model, num_classes, path, is_local, diff_classes_flag, old_num_classes)

class EfficientNetb4(ClassificationModel):
    def __init__(self, num_classes=10, path='', is_local=False, diff_classes_flag=False, old_num_classes=10):
        pretrained = False if is_local else True
        model = torch.hub.load('narumiruna/efficientnet-pytorch', 'efficientnet_b4', pretrained=pretrained)
        super().__init__(model, num_classes, path, is_local, diff_classes_flag, old_num_classes)

class EfficientNetb3(ClassificationModel):
    def __init__(self, num_classes=10, path='', is_local=False, diff_classes_flag=False, old_num_classes=10):
        pretrained = False if is_local else True
        model = torch.hub.load('narumiruna/efficientnet-pytorch', 'efficientnet_b3', pretrained=pretrained)
        super().__init__(model, num_classes, path, is_local, diff_classes_flag, old_num_classes)

class EfficientNetb0(ClassificationModel):
    def __init__(self, num_classes=10, path='', is_local=False, diff_classes_flag=False, old_num_classes=10):
        pretrained = False if is_local else True
        model = torch.hub.load('narumiruna/efficientnet-pytorch', 'efficientnet_b0', pretrained=pretrained)
        super().__init__(model, num_classes, path, is_local, diff_classes_flag, old_num_classes)

class Densenet201(ClassificationModel):
    def __init__(self, num_classes=10, path='', is_local=False, diff_classes_flag=False, old_num_classes=10):
        pretrained = False if is_local else True
        model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet201', pretrained=pretrained)
        super().__init__(model, num_classes, path, is_local, diff_classes_flag, old_num_classes)

class Densenet169(ClassificationModel):
    def __init__(self, num_classes=10, path='', is_local=False, diff_classes_flag=False, old_num_classes=10):
        pretrained = False if is_local else True
        model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet169', pretrained=pretrained)
        super().__init__(model, num_classes, path, is_local, diff_classes_flag, old_num_classes)

class Densenet161(ClassificationModel):
    def __init__(self, num_classes=10, path='', is_local=False, diff_classes_flag=False, old_num_classes=10):
        pretrained = False if is_local else True
        model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet161', pretrained=pretrained)
        super().__init__(model, num_classes, path, is_local, diff_classes_flag, old_num_classes)

class Densenet121(ClassificationModel):
    def __init__(self, num_classes=10, path='', is_local=False, diff_classes_flag=False, old_num_classes=10):
        pretrained = False if is_local else True
        model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=pretrained)
        super().__init__(model, num_classes, path, is_local, diff_classes_flag, old_num_classes)

class ResNet18(ClassificationModel):
    def __init__(self, num_classes=10, path='', is_local=False, diff_classes_flag=False, old_num_classes=10):
        pretrained = False if is_local else True
        model = vision.models.resnet18(pretrained=pretrained)
        super().__init__(model, num_classes, path, is_local, diff_classes_flag, old_num_classes)

class ResNet34(ClassificationModel):
    def __init__(self, num_classes=10, path='', is_local=False, diff_classes_flag=False, old_num_classes=10):
        pretrained = False if is_local else True
        model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet34', pretrained=pretrained)
        super().__init__(model, num_classes, path, is_local, diff_classes_flag, old_num_classes)

class ResNet50(ClassificationModel):
    def __init__(self, num_classes=10, path='', is_local=False, diff_classes_flag=False, old_num_classes=10):
        pretrained = False if is_local else True
        model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=pretrained)
        super().__init__(model, num_classes, path, is_local, diff_classes_flag, old_num_classes)

class ResNet101(ClassificationModel):
    def __init__(self, num_classes=10, path='', is_local=False, diff_classes_flag=False, old_num_classes=10):
        pretrained = False if is_local else True
        model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet101', pretrained=pretrained)
        super().__init__(model, num_classes, path, is_local, diff_classes_flag, old_num_classes)

class Resnext50_32x4d(ClassificationModel):
    def __init__(self, num_classes=10, path='', is_local=False, diff_classes_flag=False, old_num_classes=10):
        pretrained = False if is_local else True
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=pretrained)
        super().__init__(model, num_classes, path, is_local, diff_classes_flag, old_num_classes)

class Resnext101_32x8d(ClassificationModel):
    def __init__(self, num_classes=10, path='', is_local=False, diff_classes_flag=False, old_num_classes=10):
        pretrained = False if is_local else True
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext101_32x8d', pretrained=pretrained)
        super().__init__(model, num_classes, path, is_local, diff_classes_flag, old_num_classes)

class WideResnet50_2(ClassificationModel):
    def __init__(self, num_classes=10, path='', is_local=False, diff_classes_flag=False, old_num_classes=10):
        pretrained = False if is_local else True
        model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=pretrained)
        super().__init__(model, num_classes, path, is_local, diff_classes_flag, old_num_classes)

class WideResnet101_2(ClassificationModel):
    def __init__(self, num_classes=10, path='', is_local=False, diff_classes_flag=False, old_num_classes=10):
        pretrained = False if is_local else True
        model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet101_2', pretrained=pretrained)
        super().__init__(model, num_classes, path, is_local, diff_classes_flag, old_num_classes)

if __name__ == '__main__':
    model = WideResnet101_2()
