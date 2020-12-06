import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset

class CassavaDataset(Dataset):
    def __init__(self, 
                 image_paths: list, 
                 image_labels: list,
                 transforms=None,
                 image_size=None):
        assert  len(image_paths) == len(image_labels)

        self.image_paths = image_paths
        self.image_labels = image_labels
        self.transforms = transforms
        self.image_size = image_size
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        item = {}
        image_path = self.image_paths[idx]
        if isinstance(image_path, Path):
            image_path = image_path.as_posix()

        image = cv2.imread(image_path)
        if self.image_size:
            image = cv2.resize(image, self.image_size)

        if self.transforms:
            image = self.transforms(image=image)["image"]

        image = np.moveaxis(image, -1, 0)
        item["features"] = torch.from_numpy(image)
        item["targets"] = self.image_labels[idx]

        return item