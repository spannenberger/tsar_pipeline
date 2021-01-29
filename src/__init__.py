# from catalyst.dl import SupervisedRunner as Runner
from .TTASupervisedRunner import TTASupervisedRunner as Runner
from .experiment import Experiment
from .TTASupervisedRunner import TTASupervisedRunner
from .ResNextFacebook32x8d import ResNextFacebook32x8d
from .resnet18 import ResNet18
from catalyst.dl import registry
from catalyst.contrib.callbacks.confusion_matrix_logger import ConfusionMatrixCallback

from callbacks.iner_callback import InerCallback
registry.Model(ResNet18)
registry.Callback(ConfusionMatrixCallback)
registry.Model(ResNextFacebook32x8d)
