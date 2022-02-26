from catalyst.callbacks.metric import LoaderMetricCallback
from catalyst.core.callback import Callback, CallbackOrder
from catalyst.core.misc import _get_original_callback
from metrics.custom_metric import CustomMetric
from catalyst.dl import ControlFlowCallback
from catalyst.core.runner import IRunner
from catalyst.registry import Registry
from abc import abstractmethod
from pathlib import Path
from typing import Dict


class SaveMetricCallback(Callback):
    
    def __init__(self, required_metrics: Dict):
        super().__init__(CallbackOrder.External)
        self.required_metrics = required_metrics
    
    @abstractmethod
    def save_metric(self, kwargs, metric_data):
        pass

    def on_experiment_end(self, state: IRunner) -> None:
        for callback in state.callbacks:
            if callback in self.required_metrics.keys():
                current_callback = state.callbacks[callback]
                current_callback = _get_original_callback(current_callback)
                if not isinstance(current_callback, LoaderMetricCallback):
                    continue
                if issubclass(type(current_callback.metric), CustomMetric):
                    metric_data = current_callback.metric.metric_to_string()
                    self.save_metric(self.required_metrics[callback], metric_data)


@Registry
class SaveMetricInFileCallback(SaveMetricCallback):

    def save_metric(self, kwargs, metric_data):
        for arg in kwargs:
            path = Path(kwargs[arg]).absolute()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open(mode='w') as file:
                file.write(metric_data[arg])
