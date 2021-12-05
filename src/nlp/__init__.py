from .SupervisedRunner import SiameseSupervisedRunner, NLPSupervisedRunner
from .models.gpt import gpt
from .models.siamese_model import Siamese
from catalyst.registry import Registry
from .callbacks.analytics_callback import AnalyticsDistanceCallback
from .callbacks.custom_mlflow_callback import CustomMLFlowCallback
from .callbacks.duplet_metric_callback import DupletMetricCallback

Registry(Siamese)
Registry(gpt)
Registry(SiameseSupervisedRunner)
Registry(NLPSupervisedRunner)
