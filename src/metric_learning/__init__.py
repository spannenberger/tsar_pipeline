from catalyst.registry import Registry
from SupervisedRunner import MertricLearningSupervisedRunner
from models.ResNet18 import ResNet18
from models.ResNext50 import ResNext50
from models.MobilenetV3Small import MobilenetV3Small
from models.EfficientNetB0 import EfficientNetB0
from callbacks.criterion import CustomTrainCriterion
from callbacks.cmc_valid import CustomCMC
from callbacks.custom_scheduler import CustomScheduler
Registry(ResNext50)
Registry(ResNet18)
Registry(MobilenetV3Small)
Registry(EfficientNetB0)
Registry(CustomTrainCriterion)
Registry(CustomCMC)
Registry(MertricLearningSupervisedRunner)
