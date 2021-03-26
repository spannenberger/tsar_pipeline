from src_multiclass.TTASupervisedRunner import TTASupervisedRunner

# Импорт моделей
from src_multiclass.models.resnet18 import ResNet18 # converter True
from src_multiclass.models.EffNetb0 import EffNetb0 # converter onnx: False
from src_multiclass.models.EffNetb3 import EffNetb3 # converter onnx: False
from src_multiclass.models.EffNetb6 import EffNetb6 # converter onnx: False
from src_multiclass.models.densenet121 import densenet121 # converter True
from src_multiclass.models.densenet169 import densenet169 # converter True
from src_multiclass.models.densenet201 import densenet201 # converter True
from src_multiclass.models.densenet161 import densenet161 # converter True
from src_multiclass.models.resnest50 import resnest50 # False
from src_multiclass.models.resnext50_32x4d import resnext50_32x4d # converter True 
from src_multiclass.models.resnext101_32x8d import resnext101_32x8d # converter True 

# Импорт колбэков
from src_multiclass.callbacks.iner_callback import InerCallback
from src_multiclass.callbacks.new_scheduler import CustomScheduler
from src_multiclass.callbacks.logger_callbacks.mlflow_logging_callback import MLFlowloggingCallback
from src_multiclass.callbacks.logger_callbacks.tensorboard_image_logger import TensorboardImageCustomLogger

# Импорт инструментов каталиста
from catalyst.registry import Registry
from catalyst.loggers.mlflow import MLflowLogger
from catalyst.loggers.tensorboard import TensorboardLogger
from catalyst.loggers.tensorboard import TensorboardLogger
# from catalyst.callbacks.metrics.confusion_matrix import ConfusionMatrixCallback


# Инициализаця
Registry(TTASupervisedRunner)
Registry(resnext101_32x8d)
# Registry(ConfusionMatrixCallback)
