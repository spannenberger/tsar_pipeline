from multiclass.TTASupervisedRunner import TTASupervisedRunner

# Импорт моделей
from models.ResNet18 import ResNet18 # converter True
from models.ResNet34 import ResNet34
from models.ResNet50 import ResNet50
from models.ResNet101 import ResNet101
from models.EffNetb0 import EffNetb0 # converter onnx: False
from models.EffNetb3 import EffNetb3 # converter onnx: False
from models.EffNetb6 import EffNetb6 # converter onnx: False
from models.densenet121 import densenet121 # converter True
from models.densenet169 import densenet169 # converter True
from models.densenet201 import densenet201 # converter True
from models.densenet161 import densenet161 # converter True
from models.resnest50 import resnest50 # False
from models.resnext50_32x4d import resnext50_32x4d # converter True 
from models.resnext101_32x8d import resnext101_32x8d # converter True 
from models.WideResnet50_2 import WideResnet50_2
from models.WideResnet101_2 import WideResnet101_2
from models.MobilenetV2 import MobilenetV2
from models.MobilenetV3Large import MobilenetV3Large
from models.MobilenetV3Small import MobilenetV3Small
from models.ResNet18_swsl import ResNet18_swsl

# Импорт колбэков
from multiclass.multiclass_callbacks.iner_callback import InerCallback
from callbacks.custom_scheduler import CustomScheduler
from callbacks.logger_callbacks.mlflow_image_logger import MLFlowMulticlassLoggingCallback
from callbacks.logger_callbacks.tensorboard_image_logger import TensorboardMulticlassLoggingCallback

# Импорт инструментов каталиста
from catalyst.registry import Registry
from catalyst.loggers.mlflow import MLflowLogger
from catalyst.loggers.tensorboard import TensorboardLogger
from catalyst.loggers.tensorboard import TensorboardLogger
# from catalyst.callbacks.metrics.confusion_matrix import ConfusionMatrixCallback


# Инициализаця
Registry(TTASupervisedRunner)
Registry(ResNet18)
# Registry(ConfusionMatrixCallback)
