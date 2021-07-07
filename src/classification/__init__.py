from SupervisedRunner import MulticlassSupervisedRunner
from SupervisedRunner import MultilabelSupervisedRunner

# Импорт моделей
from models.ResNet18 import ResNet18  # converter True
from models.ResNet34 import ResNet34  # converter True
from models.ResNet50 import ResNet50  # converter True
from models.ResNet101 import ResNet101  # converter True
from models.EfficientNetb0 import EfficientNetb0  # converter True
from models.EfficientNetb3 import EfficientNetb3  # converter True
from models.EfficientNetb4 import EfficientNetb4  # converter True
from models.Densenet121 import Densenet121  # converter True
from models.Densenet169 import Densenet169  # converter True
from models.Densenet201 import Densenet201  # converter True
from models.Densenet161 import Densenet161  # converter True
from models.Resnext50_32x4d import Resnext50_32x4d  # converter True
from models.Resnext101_32x8d import Resnext101_32x8d  # converter True
from models.WideResnet50_2 import WideResnet50_2  # converter True
from models.WideResnet101_2 import WideResnet101_2  # converter True
from models.MobilenetV2 import MobilenetV2  # converter True
from models.MobilenetV3Large import MobilenetV3Large  # converter onnx: False
from models.MobilenetV3Small import MobilenetV3Small  # converter onnx: False
from models.ResNet18_swsl import ResNet18_swsl  # converter True

# Импорт колбэков
from callbacks.custom_scheduler import CustomScheduler
from callbacks.convert_callbacks.torchscript_save_callback import TorchscriptSaveCallback
from callbacks.convert_callbacks.onnx_save_callback import OnnxSaveCallback
from callbacks.convert_callbacks.quantization import QuantizationCallback
from callbacks.triton_config_callback import TritonConfigCreator
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
Registry(Resnext50_32x4d)
Registry(Resnext101_32x8d)
Registry(WideResnet50_2)
Registry(WideResnet101_2)
Registry(MobilenetV2)
Registry(MobilenetV3Large)
Registry(MobilenetV3Small)
Registry(ResNet18_swsl)
Registry(CustomMLflowLogger)
