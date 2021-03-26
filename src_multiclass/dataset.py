import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from src_multiclass.transform import CustomAugmentator
class CustomDataset(Dataset):
    def __init__(self,
                 image_paths: list,
                 image_labels: list,
                 valid: bool = False,
                 tta: int = 1,
                 transforms_path = None):
        assert  len(image_paths) == len(image_labels)

        self.image_paths = image_paths
        self.image_labels = image_labels
        self.transforms_path = transforms_path
        self.valid = valid
        self.tta = tta
        if valid:
            aug_mode = 'valid'
        else:
            aug_mode = 'train'
        self.transforms = CustomAugmentator().transforms(self.transforms_path, aug_mode)


    def __len__(self):
        if self.valid:
            return len(self.image_paths) * self.tta
        return len(self.image_paths)

    def __getitem__(self, idx):
        item = {}

        if self.valid:
            idx = idx // self.tta

        image_path = self.image_paths[idx]
        if isinstance(image_path, Path):
            image_path = image_path.as_posix()
        image = cv2.imread(image_path)

        if self.transforms:
            image = self.transforms(image = image)["image"]
        image = np.moveaxis(image, -1, 0)
        item["image_name"] = image_path
        item["features"] = torch.from_numpy(image)
        item["targets"] = self.image_labels[idx]

        return item
