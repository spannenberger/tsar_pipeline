from catalyst.metrics import ICallbackLoaderMetric
from abc import abstractmethod

class CustomMetric(ICallbackLoaderMetric):

    @abstractmethod
    def metric_to_string(self):
        pass