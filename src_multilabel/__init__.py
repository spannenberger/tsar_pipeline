from TTASupervisedRunner import TTASupervisedRunner

# Импорт моделей
from models.resnet18 import ResNet18
from models.EffNetb0 import EffNetb0
from models.EffNetb3 import EffNetb3
from models.EffNetb6 import EffNetb6
from models.densenet121 import densenet121
from models.resnest50 import resnest50
from models.resnext50_32x4d import resnext50_32x4d

# Импорт колбэков
from callbacks.iner_callback import InerCallback
from callbacks.new_scheduler import CustomScheduler
from catalyst.loggers.tensorboard import TensorboardLogger
from callbacks.logger_callbacks.mlflow_logging_callback import MLFlowloggingCallback
from callbacks.logger_callbacks.tensorboard_image_logger import TensorboardImageCustomLogger

# Импорт инструментов каталиста
from catalyst.registry import Registry
from catalyst.loggers.mlflow import MLflowLogger
from catalyst.loggers.tensorboard import TensorboardLogger
# from catalyst.callbacks.metrics.confusion_matrix import ConfusionMatrixCallback


# Инициализаця
Registry(TTASupervisedRunner)
Registry(resnext50_32x4d)
# Registry(ConfusionMatrixCallback)
