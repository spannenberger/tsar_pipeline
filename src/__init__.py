# from catalyst.dl import SupervisedRunner as Runner
from .TTASupervisedRunner import TTASupervisedRunner as Runner
from .experiment import Experiment
from .TTASupervisedRunner import TTASupervisedRunner
from .resnet18 import ResNet18
from catalyst.dl import registry
from catalyst.contrib.callbacks.confusion_matrix_logger import ConfusionMatrixCallback

from callbacks.iner_callback import InerCallback
from callbacks.new_scheduler import CustomScheduler

registry.Model(ResNet18)
registry.Callback(ConfusionMatrixCallback)
