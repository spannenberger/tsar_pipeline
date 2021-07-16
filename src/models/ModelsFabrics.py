from models.Classification_models import ClassificationModel
from models.MetricLearning_models import MetricLearningModel
import torchvision as vision


class ModelsFabric:

    @staticmethod
    def CreateModel(model, mode, **kwargs):
        if mode == 'Classification':
            return ModelsFabric.CreateClassification(model, **kwargs)
        if mode == 'MetricLearning':
            return ModelsFabric.CreateMetricLearning(model, **kwargs)

    @staticmethod
    def CreateClassification(model, **kwargs):
        return ClassificationModel(model, **kwargs)

    @staticmethod
    def CreateMetricLearning(model, **kwargs):
        return MetricLearningModel(model, **kwargs)


class ResNet18Fabric(ModelsFabric):

    @staticmethod
    def get_from_params(mode, **kwargs):
        model = vision.models.resnet18(pretrained=not kwargs.get("is_local", False))
        return ModelsFabric.CreateModel(model, mode, **kwargs)
