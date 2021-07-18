from SupervisedRunner import MulticlassSupervisedRunner
from SupervisedRunner import MultilabelSupervisedRunner

# Импорт моделей
from models.Models import ResNet18
from models.Models import MobilenetV3Small
from models.Models import MobilenetV3Large
from models.Models import MobilenetV2
from models.Models import EfficientNetb4
from models.Models import EfficientNetb3
from models.Models import EfficientNetb0
from models.Models import Densenet201
from models.Models import Densenet169
from models.Models import Densenet161
from models.Models import Densenet121
from models.Models import ResNet34
from models.Models import ResNet50
from models.Models import ResNet101
from models.Models import ResNext50_32x4d
from models.Models import ResNext101_32x8d
from models.Models import WideResnet50_2
from models.Models import WideResnet101_2

# Импорт колбэков
from callbacks.custom_scheduler import CustomScheduler
from callbacks.convert_callbacks.torchscript_save_callback import TorchscriptSaveCallback
from callbacks.convert_callbacks.onnx_save_callback import OnnxSaveCallback
# Multiclass
from callbacks.iner_callback import MulticlassInerCallback
from callbacks.logger_callbacks.mlflow_image_logger import MLFlowMulticlassLoggingCallback
from callbacks.logger_callbacks.tensorboard_image_logger import TensorboardMulticlassLoggingCallback
# Multilabel
from callbacks.iner_callback import MultilabelInerCallback
from callbacks.logger_callbacks.mlflow_image_logger import MLFlowMultilabelLoggingCallback
from callbacks.logger_callbacks.tensorboard_image_logger import TensorboardMultilabelLoggingCallback

from callbacks.logger_callbacks.mlflow_logger import CustomMLflowLogger
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
Registry(EfficientNetb4)
Registry(Densenet121)
Registry(Densenet169)
Registry(Densenet201)
Registry(Densenet161)
Registry(ResNext50_32x4d)
Registry(ResNext101_32x8d)
Registry(WideResnet50_2)
Registry(WideResnet101_2)
Registry(MobilenetV2)
Registry(MobilenetV3Large)
Registry(MobilenetV3Small)
Registry(CustomMLflowLogger)
