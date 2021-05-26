from typing import Any, Callable, Dict, List, Optional
import torch
import os
from torch.utils.data import Dataset
from catalyst.data.dataset.metric_learning import MetricLearningTrainDataset, QueryGalleryDataset
from torchvision.datasets import ImageFolder
import cv2


class TrainMLDataset(MetricLearningTrainDataset, ImageFolder):

    def get_labels(self) -> List[int]:
        """
        Returns:
            labels of digits
        """
        return self.targets


class ValidMLDataset(QueryGalleryDataset):

    def __init__(
        self, root_gallery: str, root_query: str, loader: Callable,
        transform: Optional[Callable] = None
    ) -> None:
        self._gallery = ImageFolder(root_gallery, loader=loader, transform=transform)
        self._query = ImageFolder(root_query, loader=loader, transform=transform)
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
        if idx < self._gallery_size:
            image, label = self._gallery[idx]
        else:
            image, label = self._query[idx - self._gallery_size]
        return {
            "features": image,
            "targets": label,
            "is_query": idx >= self._gallery_size,
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
