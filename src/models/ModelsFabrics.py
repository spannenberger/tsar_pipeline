from torch import nn
import torch
import torchvision as vision
from models.Classification_models import ClassificationModel
from models.MetricLearning_models import MetricLearningModel

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
