from typing import Any, Callable, Dict, List, Optional
from catalyst.data.dataset.metric_learning import MetricLearningTrainDataset, QueryGalleryDataset
from torchvision.datasets import ImageFolder
from transform import CustomAugmentator


# добавить трансформации сюда
class TrainMLDataset(MetricLearningTrainDataset, ImageFolder):
    def __init__(self, *args, transforms_path: str, **kwargs):
        self.transforms_path = transforms_path
        transforms = CustomAugmentator().transforms(self.transforms_path, aug_mode='train')
        super().__init__(transform=lambda x: transforms(image=x)['image'], *args, **kwargs)

    def get_labels(self) -> List[int]:
        """
        Returns:
            labels of digits
        """
        return self.targets


class ValidMLDataset(QueryGalleryDataset):
    # Возможно надо разделить аугментации галлереи и запроса
    def __init__(
        self, root_gallery: str, root_query: str, loader: Callable,
        transforms_path: str
    ) -> None:
        self.transforms_path = transforms_path
        self.transforms = CustomAugmentator().transforms(self.transforms_path, aug_mode='valid')
        self._gallery = ImageFolder(root_gallery, loader=loader,
                                    transform=lambda x: self.transforms(image=x)['image'])
        self._query = ImageFolder(root_query, loader=loader,
                                  transform=lambda x: self.transforms(image=x)['image'])
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
        if idx % 2 == 0:
            image, label = self._gallery[idx//2]
            image_name = self._gallery.imgs[idx//2][0]
        else:
            image, label = self._query[idx//2]
            image_name = self._query.imgs[idx//2][0]
        return {
            "image_name": image_name,
            "features": image,
            "targets": label,
            "is_query": idx % 2 != 0,
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
