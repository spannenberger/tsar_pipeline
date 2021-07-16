from SupervisedRunner import MulticlassSupervisedRunner
from SupervisedRunner import MultilabelSupervisedRunner

# Импорт моделей
from models.Classification_models import EfficientNetb0
from models.Classification_models import EfficientNetb3
from models.Classification_models import EfficientNetb4
from models.Classification_models import Densenet121
from models.Classification_models import Densenet201
from models.Classification_models import Densenet169
from models.Classification_models import Densenet161
from models.Classification_models import Resnext50_32x4d
from models.Classification_models import Resnext101_32x8d
from models.Classification_models import WideResnet50_2
from models.Classification_models import WideResnet101_2
from models.Classification_models import MobilenetV2
from models.Classification_models import MobilenetV3Large
from models.Classification_models import MobilenetV3Small
from models.Classification_models import ResNet50
from models.Classification_models import ResNet34
from models.Classification_models import ResNet18
from models.Classification_models import ResNet101
from models.ModelsFabrics import ResNet18Fabric

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
Registry(ResNet18Fabric)
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
Registry(CustomMLflowLogger)
