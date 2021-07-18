from torch import nn
import torch
import torchvision as vision
from models.models_fabrics import ModelsFabric


class MobilenetV3Small(ModelsFabric):

    @staticmethod
    def get_from_params(mode, **kwargs):
        model = vision.models.mobilenet_v3_small(
            pretrained=not kwargs.get("is_local", False))
        return MobilenetV3Small.create_model(model, mode, **kwargs)


class MobilenetV3Large(ModelsFabric):

    @staticmethod
    def get_from_params(mode, **kwargs):
        model = vision.models.mobilenet_v3_large(
            pretrained=not kwargs.get("is_local", False))
        return MobilenetV3Large.create_model(model, mode, **kwargs)


class MobilenetV2(ModelsFabric):

    @staticmethod
    def get_from_params(mode, **kwargs):
        model = vision.models.mobilenet_v2(
            pretrained=not kwargs.get("is_local", False))
        return MobilenetV2.create_model(model, mode, **kwargs)


class EfficientNetb4(ModelsFabric):

    @staticmethod
    def get_from_params(mode, **kwargs):
        model = torch.hub.load('narumiruna/efficientnet-pytorch',
                               'efficientnet_b4', pretrained=not kwargs.get("is_local", False))
        return EfficientNetb4.create_model(model, mode, **kwargs)


class EfficientNetb3(ModelsFabric):

    @staticmethod
    def get_from_params(mode, **kwargs):
        model = torch.hub.load('narumiruna/efficientnet-pytorch',
                               'efficientnet_b3', pretrained=not kwargs.get("is_local", False))
        return EfficientNetb3.create_model(model, mode, **kwargs)


class EfficientNetb0(ModelsFabric):

    @staticmethod
    def get_from_params(mode, **kwargs):
        model = torch.hub.load('narumiruna/efficientnet-pytorch',
                               'efficientnet_b0', pretrained=not kwargs.get("is_local", False))
        return EfficientNetb0.create_model(model, mode, **kwargs)


class Densenet201(ModelsFabric):

    @staticmethod
    def get_from_params(mode, **kwargs):
        model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet201',
                               pretrained=not kwargs.get("is_local", False))
        return Densenet201.create_model(model, mode, **kwargs)


class Densenet169(ModelsFabric):

    @staticmethod
    def get_from_params(mode, **kwargs):
        model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet169',
                               pretrained=not kwargs.get("is_local", False))
        return Densenet169.create_model(model, mode, **kwargs)


class Densenet161(ModelsFabric):

    @staticmethod
    def get_from_params(mode, **kwargs):
        model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet161',
                               pretrained=not kwargs.get("is_local", False))
        return Densenet161.create_model(model, mode, **kwargs)


class Densenet121(ModelsFabric):

    @staticmethod
    def get_from_params(mode, **kwargs):
        model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121',
                               pretrained=not kwargs.get("is_local", False))
        return Densenet121.create_model(model, mode, **kwargs)


class ResNet18(ModelsFabric):

    @staticmethod
    def get_from_params(mode, **kwargs):
        model = vision.models.resnet18(
            pretrained=not kwargs.get("is_local", False))
        return ResNet18.create_model(model, mode, **kwargs)


class ResNet34(ModelsFabric):

    @staticmethod
    def get_from_params(mode, **kwargs):
        model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet34',
                               pretrained=not kwargs.get("is_local", False))
        return ResNet34.create_model(model, mode, **kwargs)


class ResNet50(ModelsFabric):

    @staticmethod
    def get_from_params(mode, **kwargs):
        model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50',
                               pretrained=not kwargs.get("is_local", False))
        return ResNet50.create_model(model, mode, **kwargs)


class ResNet101(ModelsFabric):

    @staticmethod
    def get_from_params(mode, **kwargs):
        model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet101',
                               pretrained=not kwargs.get("is_local", False))
        return ResNet101.create_model(model, mode, **kwargs)


class ResNext50_32x4d(ModelsFabric):

    @staticmethod
    def get_from_params(mode, **kwargs):
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d',
                               pretrained=not kwargs.get("is_local", False))
        return ResNext50_32x4d.create_model(model, mode, **kwargs)


class ResNext101_32x8d(ModelsFabric):

    @staticmethod
    def get_from_params(mode, **kwargs):
        model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext101_32x8d',
                               pretrained=not kwargs.get("is_local", False))
        return ResNext101_32x8d.create_model(model, mode, **kwargs)


class WideResnet50_2(ModelsFabric):

    @staticmethod
    def get_from_params(mode, **kwargs):
        model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2',
                               pretrained=not kwargs.get("is_local", False))
        return WideResnet50_2.create_model(model, mode, **kwargs)


class WideResnet101_2(ModelsFabric):

    @staticmethod
    def get_from_params(mode, **kwargs):
        model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet101_2',
                               pretrained=not kwargs.get("is_local", False))
        return WideResnet101_2.create_model(model, mode, **kwargs)
