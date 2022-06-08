from .SupervisedRunner import SiameseSupervisedRunner, NLPSupervisedRunner

from models.models import ruGPT3Models
from models.models import SiameseGPTModels

from catalyst.registry import Registry

from callbacks.logger_callbacks.mlflow_logger import CustomMLflowLogger
from callbacks.logger_callbacks.mlflow_image_logger import MLFlowNLPLoggingCallback
from .callbacks.save_model import SaveModelWithConfigCallback

Registry(MLFlowNLPLoggingCallback)
Registry(CustomMLflowLogger)
Registry(SiameseGPTModels)
Registry(ruGPT3Models)
Registry(SiameseSupervisedRunner)
Registry(NLPSupervisedRunner)
