from torch import nn
import torch
import torchvision as vision
from models.classification_models import ClassificationModel
from models.metric_learning_models import MetricLearningModel


class ModelsFabric:

    @staticmethod
    def create_model(model, mode, fabric=None, **kwargs):
        if fabric is None:
            fabric = ModelsFabric
        if mode == 'Classification':
            return fabric.create_classification(model, **kwargs)
        if mode == 'MetricLearning':
            return fabric.create_metric_learning(model, **kwargs)

    @staticmethod
    def replace_classifier(model, num_classes):
        classifier_name, old_classifier = model.backbone._modules.popitem()
        if isinstance(old_classifier, nn.Sequential):
            input_shape = old_classifier[-1].in_features
            old_classifier[-1] = nn.Linear(input_shape, num_classes)

        elif isinstance(old_classifier, nn.Linear):
            input_shape = old_classifier.in_features
            old_classifier = nn.Linear(input_shape, num_classes)
        else:
            raise Exception("Uknown type of classifier {}".format(type(old_classifier)))
        model.backbone.add_module(classifier_name, old_classifier)

    @staticmethod
    def create_classification(model, **kwargs):
        path = kwargs.pop('path', '')
        diff_classes_flag = kwargs.pop('diff_classes_flag', False)
        is_local = kwargs.pop('is_local', False)
        old_num_classes = kwargs.pop('old_num_classes', 10)
        num_classes = kwargs.pop('num_classes', 10)

        model = ClassificationModel(model)
        if is_local:
            if diff_classes_flag:
                ModelsFabric.replace_classifier(model, old_num_classes)
                model.load_state_dict(torch.load(path)['model_state_dict'])
                ModelsFabric.replace_classifier(model, num_classes)
            else:
                ModelsFabric.replace_classifier(model, num_classes)
                model.load_state_dict(torch.load(path)['model_state_dict'])
        else:
            ModelsFabric.replace_classifier(model, num_classes)
        return model

    @staticmethod
    def create_metric_learning(model, **kwargs):
        path = kwargs.pop('path', '')
        is_local = kwargs.pop('is_local', False)
        model = torch.nn.Sequential(
            *(list(model.children())[:-1]))
        model = MetricLearningModel(model)
        if is_local:
            model.load_state_dict(torch.load(path)['model_state_dict'])
        return model
