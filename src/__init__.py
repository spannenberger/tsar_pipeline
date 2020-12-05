from catalyst.dl import SupervisedRunner as Runner
from .experiment import Experiment
from .resnet18 import ResNet18
from catalyst.dl import registry

registry.Model(ResNet18)