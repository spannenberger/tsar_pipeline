from catalyst.core import IRunner
from catalyst.runners import SupervisedConfigRunner
from collections import OrderedDict
from catalyst.contrib import datasets
import os
from catalyst.data.transforms import Compose, Normalize, ToTensor
from catalyst import data


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
        transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        datasetss = OrderedDict()
        train_dataset = datasets.MnistMLDataset(
            root=os.getcwd(), download=True, transform=transforms)
        sampler = data.BalanceBatchSampler(labels=train_dataset.get_labels(), p=5, k=10)
        valid_dataset = datasets.MnistQGDataset(
            root=os.getcwd(), transform=transforms, gallery_fraq=0.2)

        datasetss["train"] = {'dataset': train_dataset,
                              'sampler': sampler,
                              'batch_size': sampler.batch_size
                              }
        datasetss["valid"] = {'dataset': valid_dataset}

        return datasetss


class MertricLearningSupervisedRunner(MertricLearningRunner, SupervisedConfigRunner):
    pass
