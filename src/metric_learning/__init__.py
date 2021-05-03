from catalyst.registry import Registry
from SupervisedRunner import MertricLearningSupervisedRunner
from models.resnet50 import ResNet50
from callbacks.criterion import CustomTrainCriterion
from callbacks.cmc_valid import CustomCMC
from callbacks.custom_scheduler import CustomScheduler
Registry(ResNet50)
Registry(CustomTrainCriterion)
Registry(CustomCMC)
Registry(MertricLearningSupervisedRunner)
