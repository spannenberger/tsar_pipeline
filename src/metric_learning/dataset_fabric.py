from collections import OrderedDict
from pathlib import Path
from catalyst import data
from datasets.datasets_fabric import DatasetsFabric

from dataset import LMDBTrainMLDataset, LMDBValidMLDataset, TrainMLDataset, ValidMLDataset


class MetricLearningDatasetCreator(DatasetsFabric):
    datasets = {
        "LMDB": {
            "train": LMDBTrainMLDataset,
            "valid": LMDBValidMLDataset
        },
        "Classic": {
            "train": TrainMLDataset,
            "valid": ValidMLDataset
        },
    }

    def get_config_params(config_params):
        dataset_params = {}
        dataset_params["dataset_path"] = Path(config_params["dataset_path"])

        dataset_params = {
            "transform_path": Path(config_params["transform_path"]),
            "train_path": Path(dataset_params["dataset_path"] / config_params["train_path"]),
            "base_path": Path(dataset_params["dataset_path"] / config_params["base_path"]),
            "val_path": Path(dataset_params["dataset_path"] / config_params["val_path"]),
            "is_check": config_params["is_check"]
        }

        return dataset_params

    @classmethod
    def create_datasets(cls, config_params, **kwargs):

        datasets = OrderedDict()
        dataset_params = cls.get_config_params(config_params)
        train_dataset = cls.datasets[config_params["mode"]]["train"](
            str(dataset_params["train_path"]),
            transforms_path=str(dataset_params["transform_path"]),
        )

        sampler = data.BalanceBatchSampler(labels=train_dataset.get_labels(), p=2, k=10)

        valid_dataset = cls.datasets[config_params["mode"]]["valid"](
            str(dataset_params["base_path"]),
            str(dataset_params["val_path"]),
            transforms_path=str(dataset_params["transform_path"]),
            is_check=dataset_params["is_check"],
        )

        datasets["train"] = {"dataset": train_dataset, "sampler": sampler}
        datasets["valid"] = {"dataset": valid_dataset}

        return datasets
