from catalyst.registry import Registry
from SupervisedRunner import MertricLearningSupervisedRunner

from models.ResNet18 import ResNet18
from models.ResNext50 import ResNext50
from models.MobilenetV3Small import MobilenetV3Small

from callbacks.convert_callbacks.onnx_save_callback import OnnxSaveCallback
from callbacks.convert_callbacks.torchscript_save_callback import TorchscriptSaveCallback
from callbacks.criterion import CustomTrainCriterion
from callbacks.cmc_valid import CustomCMC
from callbacks.custom_scheduler import CustomScheduler

Registry(ResNext50)
Registry(ResNet18)
Registry(MobilenetV3Small)
Registry(CustomTrainCriterion)
Registry(CustomCMC)
Registry(MertricLearningSupervisedRunner)
