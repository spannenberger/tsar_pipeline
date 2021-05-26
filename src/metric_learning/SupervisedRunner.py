from catalyst.core import IRunner
from catalyst.runners import SupervisedConfigRunner
from collections import OrderedDict
from catalyst.contrib import datasets
import os
from catalyst import data
import dataset
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensor


class MertricLearningRunner(IRunner):
    """Кастомный runner нашего эксперимента"""

    def handle_batch(self, batch) -> None:
        if self.is_train_loader:
            images, targets = batch["features"].float(), batch["targets"].long()
            features = self.model(images)
            self.batch = {"embeddings": features, "targets": targets, }
        else:
            images, targets, is_query = \
                batch["features"].float(), batch["targets"].long(), batch["is_query"].bool()
            features = self.model(images)
            self.batch = {"embeddings": features, "targets": targets, "is_query": is_query}

    def get_datasets(self, stage: str, **kwargs):
        """Работа с данными, формирование train и valid"""
        transforms = A.Compose([A.Normalize((0.1307,), (0.3081,)),
                                A.Resize(height=224, width=224), ToTensor()])
        datasetss = OrderedDict()
        train_dataset = dataset.TrainMLDataset(
            "metric_learning_dataset/train/", loader=cv2.imread, transform=lambda x: transforms(image=x)['image'])
        sampler = data.BalanceBatchSampler(labels=train_dataset.get_labels(), p=5, k=10)
        valid_dataset = dataset.ValidMLDataset("metric_learning_dataset/base/",
                                               "metric_learning_dataset/val/", loader=cv2.imread, transform=lambda x: transforms(image=x)['image'])
        print(sampler.batch_size)
        datasetss["train"] = {'dataset': train_dataset,
                              'sampler': sampler,
                              'batch_size': sampler.batch_size
                              }
        datasetss["valid"] = {'dataset': valid_dataset}

        return datasetss


class MertricLearningSupervisedRunner(MertricLearningRunner, SupervisedConfigRunner):
    pass
