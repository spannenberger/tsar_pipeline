import random
import json
from pathlib import Path
import pandas as pd
import albumentations as A
import numpy as np
from collections import OrderedDict
from catalyst.dl import ConfigExperiment
from .dataset import CassavaDataset
from sklearn.model_selection import train_test_split


class Experiment(ConfigExperiment):
    # @staticmethod
    # def get_transforms(mode: str, **kwargs):
    #     assert mode in ["train", "val"]

    #     if mode == "train":
    #         return A.Normalize()
    #     else:
    #         return A.Normalize()

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()
        data_params = self.stages_config[stage]["data_params"]

        image_size = data_params.get("image_size")
        if image_size is not None:
            assert len(image_size) == 2
            image_size = tuple(image_size)

        train_dir = Path(data_params["train_dir"])
        metadata_path = train_dir.joinpath(data_params["train_meta"])
        images_dir = train_dir.joinpath(data_params["image_dir"])
        
        train_meta = pd.read_csv(metadata_path)
        train_meta["label"] = train_meta["label"].astype(np.int64)
        
        image_paths = [images_dir.joinpath(i) for i in train_meta["image_id"]]
        labels = train_meta["label"].tolist()

        image_paths_train, image_paths_val, \
        labels_train, labels_val = train_test_split(image_paths, labels,
                                                    stratify=labels,
                                                    test_size=data_params["valid_size"])

        datasets["train"] = CassavaDataset(image_paths_train,
                                           labels_train,
                                           transforms=self.get_transforms(stage, "train"))

        datasets["valid"] = CassavaDataset(image_paths_val,
                                          labels_val,
                                          transforms=self.get_transforms(stage, "valid"))

                                           
        return datasets