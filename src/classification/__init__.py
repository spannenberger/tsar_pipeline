from SupervisedRunner import MulticlassSupervisedRunner
from SupervisedRunner import MultilabelSupervisedRunner

# Импорт моделей
from models.ResNet18 import ResNet18  # converter True
from models.ResNet34 import ResNet34
from models.ResNet50 import ResNet50
from models.ResNet101 import ResNet101
from models.EfficientNetb0 import EfficientNetb0  # converter onnx: False
from models.EfficientNetb3 import EfficientNetb3  # converter onnx: False
from models.EfficientNetb6 import EfficientNetb6  # converter onnx: False
from models.Densenet121 import densenet121  # converter True
from models.Densenet169 import Densenet169  # converter True
from models.Densenet201 import Densenet201  # converter True
from models.Densenet161 import Densenet161  # converter True
from models.Resnest50 import Resnest50  # False
from models.Resnext50_32x4d import Resnext50_32x4d  # converter True
from models.Resnext101_32x8d import Resnext101_32x8d  # converter True
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
Registry(EfficientNetb0)
Registry(EfficientNetb3)
Registry(EfficientNetb6)
Registry(Densenet121)
Registry(Densenet169)
Registry(Densenet201)
Registry(Densenet161)
Registry(Resnest50)
Registry(Resnext50_32x4d)
Registry(Resnext101_32x8d)
Registry(WideResnet50_2)
Registry(WideResnet101_2)
Registry(MobilenetV2)
Registry(MobilenetV3Large)
Registry(MobilenetV3Small)
Registry(ResNet18_swsl)
