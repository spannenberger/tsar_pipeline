from TTASupervisedRunner import TTASupervisedRunner
from resnet18 import ResNet18
from catalyst.registry import Registry
from catalyst.callbacks.metrics.confusion_matrix import ConfusionMatrixCallback
from callbacks.iner_callback import InerCallback
from callbacks.new_scheduler import CustomScheduler
from catalyst.loggers.tensorboard import TensorboardLogger
# from callbacks.metric_callbacks.recall_callback import RecallMetric

from callbacks.logger_callbacks.mlflow_logging_callback import MLFlowloggingCallback
from callbacks.logger_callbacks.tensorboard_image_logger import TensorboardImageCustomLogger
from catalyst.loggers.mlflow import MLflowLogger


Registry(TTASupervisedRunner)
Registry(ResNet18)
Registry(ConfusionMatrixCallback)
