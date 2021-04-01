from SupervisedRunner import MulticlassSupervisedRunner
from SupervisedRunner import MultilabelSupervisedRunner

# Импорт моделей
from models.ResNet18 import ResNet18  # converter True
from models.ResNet34 import ResNet34
from models.ResNet50 import ResNet50
from models.ResNet101 import ResNet101
from models.EffNetb0 import EffNetb0  # converter onnx: False
from models.EffNetb3 import EffNetb3  # converter onnx: False
from models.EffNetb6 import EffNetb6  # converter onnx: False
from models.densenet121 import densenet121  # converter True
from models.densenet169 import densenet169  # converter True
from models.densenet201 import densenet201  # converter True
from models.densenet161 import densenet161  # converter True
from models.resnest50 import resnest50  # False
from models.resnext50_32x4d import resnext50_32x4d  # converter True
from models.resnext101_32x8d import resnext101_32x8d  # converter True
from models.WideResnet50_2 import WideResnet50_2
from models.WideResnet101_2 import WideResnet101_2
from models.MobilenetV2 import MobilenetV2
from models.MobilenetV3Large import MobilenetV3Large
from models.MobilenetV3Small import MobilenetV3Small
from models.ResNet18_swsl import ResNet18_swsl

# Импорт колбэков
from callbacks.custom_scheduler import CustomScheduler
# Multiclass
from callbacks.iner_callback import MulticlassInerCallback
from callbacks.logger_callbacks.mlflow_image_logger import MLFlowMulticlassLoggingCallback
from callbacks.logger_callbacks.tensorboard_image_logger import TensorboardMulticlassLoggingCallback
# Multilabel
from callbacks.iner_callback import MultilabelInerCallback
from callbacks.logger_callbacks.mlflow_image_logger import MLFlowMultilabelLoggingCallback
from callbacks.logger_callbacks.tensorboard_image_logger import TensorboardMultilabelLoggingCallback

# Импорт инструментов каталиста
from catalyst.registry import Registry
from catalyst.loggers.mlflow import MLflowLogger
from catalyst.loggers.tensorboard import TensorboardLogger
from catalyst.loggers.tensorboard import TensorboardLogger


# Инициализаця
Registry(MulticlassSupervisedRunner)
Registry(MultilabelSupervisedRunner)
Registry(ResNet18)
Registry(ResNet34)
Registry(ResNet50)
Registry(ResNet101)
Registry(EffNetb0)
Registry(EffNetb3)
Registry(EffNetb6)
Registry(densenet121)
Registry(densenet169)
Registry(densenet201)
Registry(densenet161)
Registry(resnest50)
Registry(resnext50_32x4d)
Registry(resnext101_32x8d)
Registry(WideResnet50_2)
Registry(WideResnet101_2)
Registry(MobilenetV2)
Registry(MobilenetV3Large)
Registry(MobilenetV3Small)
Registry(ResNet18_swsl)
