from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Dict


class DatasetsFabric(ABC):
    datasets: Dict[str, Dict[str, Dataset]]

    @abstractmethod
    def create_datasets(cls, config_params, **kwargs):
        pass

    @abstractmethod
    def get_config_params(config_params):
        pass
