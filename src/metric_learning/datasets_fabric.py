from collections import OrderedDict
from catalyst import data
from pathlib import Path
import dataset
import cv2


class DatasetFabric:
    @classmethod
    def create_dataset(cls, config_params, **kwargs):
        if config_params["mode"] == "LMDB":
            return cls.create_lmdb_runner(config_params, **kwargs)
        if config_params["mode"] == "Classic":
            return cls.create_classic_runner(config_params, **kwargs)

    @staticmethod
    def parse_config_params(config_params):
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

    @staticmethod
    def create_lmdb_runner(config_params, **kwargs):
        datasets = OrderedDict()
        dataset_params = DatasetFabric.parse_config_params(config_params)
        train_dataset = dataset.LMDBTrainMLDataset(
            str(dataset_params["train_path"]),
            transforms_path=str(dataset_params["transform_path"]),
        )

        valid_dataset = dataset.LMDBValidMLDataset(
            str(dataset_params["base_path"]),
            str(dataset_params["val_path"]),
            transforms_path=str(dataset_params["transform_path"]),
            is_check=dataset_params["is_check"],
        )

        datasets["train"] = {"dataset": train_dataset}
        datasets["valid"] = {"dataset": valid_dataset}

        return datasets

    @staticmethod
    def create_classic_runner(config_params, **kwargs):
        datasets = OrderedDict()
        dataset_params = DatasetFabric.parse_config_params(config_params)
        train_dataset = dataset.TrainMLDataset(
            dataset_params["train_path"],
            loader=lambda image: cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB),
            transforms_path=dataset_params["transform_path"],
        )
        sampler = data.BalanceBatchSampler(labels=train_dataset.get_labels(), p=2, k=10)
        valid_dataset = dataset.ValidMLDataset(
            dataset_params["base_path"],
            dataset_params["val_path"],
            loader=lambda image: cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB),
            transforms_path=dataset_params["transform_path"],
            is_check=dataset_params["is_check"],
        )

        datasets["train"] = {"dataset": train_dataset, "sampler": sampler}
        datasets["valid"] = {"dataset": valid_dataset}
        return datasets
