import random
import json
from pathlib import Path
import pandas as pd
import albumentations as A
import numpy as np
from collections import OrderedDict
from catalyst.dl import ConfigExperiment
from .dataset import CassavaDataset
from sklearn.model_selection import train_test_split, StratifiedKFold


class Experiment(ConfigExperiment):
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
        tta = data_params.get('tta', 1)
        if not data_params.get("num_folds", 0) > 0:
            image_paths_train, image_paths_val, \
            labels_train, labels_val = train_test_split(image_paths, labels,
                                                        stratify=labels,
                                                        test_size=data_params["valid_size"])
        else:
            kfold = StratifiedKFold(n_splits=data_params["num_folds"],
                                    shuffle=True,
                                    random_state=self.initial_seed)

            image_paths_train, image_paths_val = [], []
            labels_train, labels_val = [], []

            for fold_index, (train_index, val_index) in enumerate(kfold.split(image_paths, labels)):
                if fold_index != data_params["fold_index"]:
                    continue

                for i in train_index:
                    image_paths_train.append(image_paths[i])
                    labels_train.append(labels[i])

                for i in val_index:
                    image_paths_val.append(image_paths[i])
                    labels_val.append(labels[i])


        datasets["train"] = CassavaDataset(image_paths_train,
                                           labels_train,
                                           transforms=self.get_transforms(stage, "train"))

        datasets["valid"] = CassavaDataset(image_paths_val,
                                          labels_val,
                                          transforms=self.get_transforms(stage, "valid"),
                                          valid = True,
                                          tta = tta)


        return datasets
