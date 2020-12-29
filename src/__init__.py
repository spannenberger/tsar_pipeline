from catalyst.dl import SupervisedRunner as Runner
from .experiment import Experiment
from .resnet18 import ResNet18
from catalyst.dl import registry
from catalyst.contrib.callbacks.confusion_matrix_logger import ConfusionMatrixCallback

from callbacks.iner_callback import InerCallback

registry.Callback(ConfusionMatrixCallback)
registry.Model(ResNet18)
#registry.Model(VGG)
#registry.Model(DenseNet121)