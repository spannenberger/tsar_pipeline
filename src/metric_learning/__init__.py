from catalyst.registry import Registry
from SupervisedRunner import MertricLearningSupervisedRunner

from models.models import ResNet18
from models.models import MobilenetV3Large
from models.models import MobilenetV3Small
from models.models import MobilenetV2
from models.models import ResNet34
from models.models import ResNet50
from models.models import ResNet101
from models.models import Densenet121 
from models.models import Densenet161 
from models.models import Densenet169
from models.models import Densenet201
from models.models import ResNext50_32x4d
from models.models import ResNext101_32x8d
from models.models import WideResnet50_2
from models.models import WideResnet101_2
from models.models import EfficientNetb0 
from models.models import EfficientNetb4
from models.models import EfficientNetb3


from callbacks.logger_callbacks.mlflow_image_logger import MLFlowMetricLearningCallback
from callbacks.logger_callbacks.tensorboard_image_logger import TensorboardMetricLearningCallback
from catalyst.loggers.tensorboard import TensorboardLogger
from callbacks.convert_callbacks.onnx_save_callback import OnnxSaveCallback
from callbacks.convert_callbacks.torchscript_save_callback import TorchscriptSaveCallback
from callbacks.criterion import CustomTrainCriterion
from callbacks.cmc_valid import CustomCMC
from callbacks.iner_callback import MLInerCallback
from callbacks.custom_scheduler import CustomScheduler

Registry(EfficientNetb0)
Registry(EfficientNetb4)
Registry(EfficientNetb3)
Registry(ResNet34)
Registry(ResNet50)
Registry(ResNet101)
Registry(ResNext50_32x4d)
Registry(ResNext101_32x8d)
Registry(Densenet121)
Registry(Densenet161)
Registry(Densenet169)
Registry(Densenet201)
Registry(WideResnet50_2)
Registry(WideResnet101_2)
Registry(MobilenetV2)
Registry(MobilenetV3Large)
Registry(ResNet18)
Registry(MobilenetV3Small)
Registry(CustomTrainCriterion)
Registry(CustomCMC)
Registry(MertricLearningSupervisedRunner)
