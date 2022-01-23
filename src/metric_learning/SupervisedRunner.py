from catalyst.runners import SupervisedConfigRunner
from collections import OrderedDict
from catalyst.core import IRunner
from catalyst import data
from pathlib import Path
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
        datasets = OrderedDict()
        data_params = self._stage_config[stage]["data"]
        dataset_path = Path(data_params["dataset_path"])
        transform_path = Path(data_params["transform_path"])
        train_path = Path(dataset_path/data_params["train_path"])
        base_path = Path(dataset_path/data_params["base_path"])
        val_path = Path(dataset_path/data_params["val_path"])
        mode = data_params["mode"]
        if mode == "LMDB":
            train_dataset = dataset.LMDBTrainMLDataset(
                str(train_path), transforms_path=transform_path)

            valid_dataset = dataset.LMDBValidMLDataset(str(base_path),
                                                str(val_path),
                                                transforms_path=transform_path,
                                                is_check=self.hparams['args'].get('check', False))

            datasets["train"] = {'dataset': train_dataset}
            datasets["valid"] = {'dataset': valid_dataset}
        else:
            train_dataset = dataset.TrainMLDataset(
                train_path, loader=lambda image : cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB), transforms_path=transform_path)
            sampler = data.BalanceBatchSampler(labels=train_dataset.get_labels(), p=2, k=10)
            valid_dataset = dataset.ValidMLDataset(base_path,
                                                val_path,
                                                loader=lambda image : cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB),
                                                transforms_path=transform_path,
                                                is_check=self.hparams['args'].get('check', False))
            datasets["train"] = {'dataset': train_dataset, 'sampler': sampler}
            datasets["valid"] = {'dataset': valid_dataset}
        return datasets


class MertricLearningSupervisedRunner(MertricLearningRunner, SupervisedConfigRunner):
    pass
