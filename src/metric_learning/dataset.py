from catalyst.data.dataset.metric_learning import (
    MetricLearningTrainDataset,
    QueryGalleryDataset,
)
from typing import Any, Dict, List
import cv2
from torchvision.datasets import ImageFolder
from utils.transform import CustomAugmentator
from torch.utils.data import Dataset
from functools import lru_cache
from tqdm import tqdm
import pyxis as px


class TrainMLDataset(MetricLearningTrainDataset, ImageFolder):
    def __init__(self, *args, transforms_path: str, **kwargs):
        self.transforms_path = transforms_path
        transforms = CustomAugmentator().transforms(
            self.transforms_path, aug_mode="train"
        )
        super().__init__(
            transform=lambda x: transforms(image=x)["image"],
            loader=lambda image: cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB),
            *args,
            **kwargs
        )

    def get_labels(self) -> List[int]:
        """
        Returns:
            labels of digits
        """
        return self.targets


class ValidMLDataset(QueryGalleryDataset):
    # Возможно надо разделить аугментации галлереи и запроса
    def __init__(
        self,
        root_gallery: str,
        root_query: str,
        transforms_path: str,
        is_check: bool = False,
    ) -> None:
        self.is_check = is_check
        self.transforms_path = transforms_path
        self.transforms = CustomAugmentator().transforms(
            self.transforms_path, aug_mode="valid"
        )
        self._gallery = ImageFolder(
            root_gallery,
            loader=lambda image: cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB),
            transform=lambda x: self.transforms(image=x)["image"],
        )
        self._query = ImageFolder(
            root_query,
            loader=lambda image: cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB),
            transform=lambda x: self.transforms(image=x)["image"],
        )
        self._gallery_size = len(self._gallery)
        self._query_size = len(self._query)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item method for dataset
        Args:
            idx: index of the object
        Returns:
            Dict with features, targets and is_query flag
        """
        if self.is_check:
            if idx % 2 == 0:
                image, label = self._gallery[idx // 2]
                image_name = self._gallery.imgs[idx // 2][0]
            else:
                image, label = self._query[idx // 2]
                image_name = self._query.imgs[idx // 2][0]
        else:
            if idx < self._gallery_size:
                image, label = self._gallery[idx]
                image_name = self._gallery.imgs[idx][0]
            else:
                image, label = self._query[idx - self._gallery_size]
                image_name = self._query.imgs[idx - self._gallery_size][0]
        return {
            "image_name": image_name,
            "features": image,
            "targets": label,
            "is_query": idx % 2 == 0 if self.is_check else idx >= self._gallery_size,
        }

    def __len__(self) -> int:
        """Length"""
        return self._gallery_size + self._query_size

    @property
    def gallery_size(self) -> int:
        """Query Gallery dataset should have gallery_size property"""
        return self._gallery_size

    @property
    def query_size(self) -> int:
        """Query Gallery dataset should have query_size property"""
        return self._query_size


class LMDBTrainMLDataset(MetricLearningTrainDataset, Dataset):
    def __init__(self, data_folder, transforms_path: str, aug_mode: str = "train"):
        self.data_folder = data_folder
        self.transforms_path = transforms_path
        self.transforms = CustomAugmentator().transforms(
            self.transforms_path, aug_mode=aug_mode
        )
        db = px.Reader(dirpath=self.data_folder)
        self.size = len(db)
        db.close()

    def __getitem__(self, idx):
        if not hasattr(self, "db"):
            self.db = px.Reader(dirpath=self.data_folder)

        item = {}
        tmp = self.db[idx]
        image = tmp["image"]
        label = tmp["target"]
        image_name = str(tmp["image_name"])

        if self.transforms:
            image = self.transforms(image=image)["image"]

        item["features"] = image
        item["targets"] = label[0]
        item["image_name"] = image_name
        return item

    def __len__(self):
        return self.size

    @lru_cache(maxsize=30)
    def get_labels(self) -> List[int]:
        """
        Returns:
            labels of digits
        """
        db = px.Reader(dirpath=self.data_folder)

        full_targets = []
        for i in tqdm(db):
            full_targets.append(i["target"][0])
        db.close()
        return full_targets


class LMDBValidMLDataset(QueryGalleryDataset):
    def __init__(
        self,
        root_gallery: str,
        root_query: str,
        transforms_path: str,
        is_check: bool = False,
    ) -> None:
        self.is_check = is_check

        self._gallery = LMDBTrainMLDataset(
            root_gallery, transforms_path=transforms_path, aug_mode="valid"
        )
        self._query = LMDBTrainMLDataset(
            root_query, transforms_path=transforms_path, aug_mode="valid"
        )

        self._gallery_size = len(self._gallery)
        self._query_size = len(self._query)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item method for dataset
        Args:
            idx: index of the object
        Returns:
            Dict with features, targets and is_query flag
        """
        if self.is_check:
            if idx % 2 == 0:
                tmp = self._gallery[idx // 2]
                image = tmp["features"]
                label = tmp["targets"]
                image_name = tmp["image_name"]
            else:
                tmp = self._query[idx // 2]
                image = tmp["features"]
                label = tmp["targets"]
                image_name = tmp["image_name"]
        else:
            if idx < self._gallery_size:
                tmp = self._gallery[idx]
                image = tmp["features"]
                label = tmp["targets"]
                image_name = tmp["image_name"]
            else:
                tmp = self._query[idx - self._gallery_size]
                image = tmp["features"]
                label = tmp["targets"]
                image_name = tmp["image_name"]
        return {
            "features": image,
            "targets": label,
            "image_name": image_name,
            "is_query": idx % 2 == 0 if self.is_check else idx >= self._gallery_size,
        }

    def __len__(self) -> int:
        """Length"""
        return self._gallery_size + self._query_size

    @property
    def gallery_size(self) -> int:
        """Query Gallery dataset should have gallery_size property"""
        return self._gallery_size

    @property
    def query_size(self) -> int:
        """Query Gallery dataset should have query_size property"""
        return self._query_size
