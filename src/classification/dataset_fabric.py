from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from dataset import CustomDataset
from datasets.datasets_fabric import DatasetsFabric



class ClassificationDatasetCreator(DatasetsFabric):
    datasets = {
        "Classic": {
            "train": CustomDataset,
            "valid": CustomDataset
        },
    }
    @classmethod
    def get_config_params(cls, config_params):
        image_size = config_params.get("image_size")
        if image_size is not None:
            assert len(image_size) == 2
            image_size = tuple(image_size)

        train_dir = Path(config_params["train_dir"])
        metadata_path = train_dir / config_params["train_meta"]
        test_path = train_dir / config_params["test_meta"]

        train_images_dir = train_dir / config_params["train_image_dir"]
        test_images_dir = train_dir / config_params["test_image_dir"]

        train_meta = pd.read_csv(metadata_path)
        test_meta = pd.read_csv(test_path)

        if config_params["task_mode"] == "Multilabel":
            return cls.create_multilabel_params(train_images_dir, test_images_dir, train_meta, test_meta)

        elif config_params["task_mode"] == "Multiclass":
            return cls.create_multiclass_params(train_images_dir, test_images_dir, train_meta, test_meta)
        else:
            raise Exception(
                f"Unknown task mode for dataset creator: {config_params['task_mode']}"
            )

    @classmethod
    def create_datasets(cls, config_params, **kwargs):
        datasets = OrderedDict()
        image_paths_train, labels_train, image_paths_val, labels_val = cls.get_config_params(config_params)

        train_dataset = cls.datasets[config_params["mode"]]["train"](
            image_paths_train,
            labels_train,
            transform_path=config_params["transform_path"],
        )

        valid_dataset = cls.datasets[config_params["mode"]]["valid"](
            image_paths_val,
            labels_val,
            transform_path=config_params["transform_path"],
            valid=True,
        )

        datasets["train"] = {"dataset": train_dataset}
        datasets["valid"] = {"dataset": valid_dataset}
        return datasets


    @staticmethod
    def create_multiclass_params(train_images_dir, test_images_dir, train_meta, test_meta):
        """Creating train, val and test metadata for multiclass classification"""

        train_meta["label"] = train_meta["label"].astype(np.int64)
        test_meta["label"] = test_meta["label"].astype(np.int64)

        train_image_paths = [train_images_dir / i for i in train_meta["image_path"]]
        test_image_paths = [test_images_dir / i for i in test_meta["image_path"]]

        train_labels = train_meta["label"].tolist()
        test_labels = test_meta["label"].tolist()

        image_paths_train = train_image_paths
        labels_train = train_labels
        image_paths_val = test_image_paths
        labels_val = test_labels

        return image_paths_train, labels_train, image_paths_val, labels_val

    @staticmethod
    def create_multilabel_params(train_images_dir, test_images_dir, train_meta, test_meta):
        """Creating train, val and test metadata for multilabel classification"""

        train_meta['label'] = [i[0] for i in zip(train_meta.iloc[:, 1:].values)]
        test_meta['label'] = [i[0] for i in zip(test_meta.iloc[:, 1:].values)]

        train_image_paths = [train_images_dir / i for i in train_meta["path"]]
        test_image_paths = [test_images_dir / i for i in test_meta["path"]]

        train_labels = train_meta["label"].tolist()
        train_labels = torch.Tensor(train_labels)
        train_labels = train_labels.type(torch.DoubleTensor)

        test_labels = test_meta["label"].tolist()
        test_labels = torch.Tensor(test_labels)
        test_labels = test_labels.type(torch.DoubleTensor)

        image_paths_train = train_image_paths
        labels_train = train_labels
        image_paths_val = test_image_paths
        labels_val = test_labels
        
        return image_paths_train, labels_train, image_paths_val, labels_val