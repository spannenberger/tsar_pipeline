from .SupervisedRunner import SiameseSupervisedRunner, NLPSupervisedRunner
from .models.gpt import gpt
from .models.siamese_model import Siamese
from catalyst.registry import Registry
from .callbacks.analytics_callback import AnalyticsDistanceCallback
from .callbacks.custom_mlflow_callback import CustomMLFlowCallback
from callbacks.logger_callbacks.mlflow_logger import CustomMLflowLogger
from callbacks.logger_callbacks.mlflow_image_logger import MLFlowNLPLoggingCallback
from .callbacks.save_model import SaveModelWithConfigCallback


Registry(MLFlowNLPLoggingCallback)
Registry(CustomMLflowLogger)
Registry(Siamese)
Registry(gpt)
Registry(SiameseSupervisedRunner)
Registry(NLPSupervisedRunner)
