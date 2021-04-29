import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from transform import CustomAugmentator


class CustomDataset(Dataset):
    """
    Работа с данными и применение к ним указанных в конфиге аугментаций
    """

    def __init__(self,
                 image_paths: list,
                 image_labels: list,
                 valid: bool = False,
                 transforms_path=None):
        assert len(image_paths) == len(image_labels)
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.transforms_path = transforms_path
        self.valid = valid
        if valid:
            aug_mode = 'valid'
        else:
            aug_mode = 'train'
        self.transforms = CustomAugmentator().transforms(self.transforms_path, aug_mode)

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        item = {}
        idx = idx

        image_path = self.image_paths[idx]
        if isinstance(image_path, Path):
            image_path = image_path.as_posix()
        image = cv2.imread(image_path)
        if self.transforms:
            image = self.transforms(image=image)["image"]

        image = np.moveaxis(image, -1, 0)
        item["image_name"] = image_path
        item["features"] = torch.from_numpy(image)
        item["targets"] = self.image_labels[idx]
        return item
