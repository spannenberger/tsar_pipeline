from catalyst.core import IRunner
from catalyst.runners import SupervisedConfigRunner
from collections import OrderedDict
from catalyst import data
import dataset
import cv2


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
            self.batch = {"embeddings": features, "targets": targets,
                          "is_query": is_query, "image_name": batch["image_name"]}

    def get_datasets(self, stage: str, **kwargs):
        """Работа с данными, формирование train и valid"""
        datasetss = OrderedDict()
        data_params = self._stage_config[stage]["data"]
        dataset_path = data_params["dataset_path"]
        trainsforms_path = data_params["transforms_path"]
        train_path = f'{dataset_path}/{data_params["train_path"]}/'
        base_path = f'{dataset_path}/{data_params["base_path"]}/'
        val_path = f'{dataset_path}/{data_params["val_path"]}/'
        train_dataset = dataset.TrainMLDataset(
            train_path, loader=cv2.imread, transforms_path=trainsforms_path)
        sampler = data.BalanceBatchSampler(labels=train_dataset.get_labels(), p=5, k=10)
        valid_dataset = dataset.ValidMLDataset(base_path,
                                               val_path, loader=cv2.imread, transforms_path=trainsforms_path)
        datasetss["train"] = {'dataset': train_dataset, 'sampler': sampler}
        datasetss["valid"] = {'dataset': valid_dataset}

        return datasetss


class MertricLearningSupervisedRunner(MertricLearningRunner, SupervisedConfigRunner):
    pass
