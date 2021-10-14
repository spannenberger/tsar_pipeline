from SupervisedRunner import ruBertNLPRunner
from model import Bert
from catalyst.registry import Registry
from transformers import AdamW
from nlp.callbacks.nlp_mlflow_logger_callback import NLPMlflowLoggerCallback
from catalyst.loggers.mlflow import MLflowLogger
from callbacks.nlp_mlflow_logger_callback import NLPMlflowLoggerCallback
from callbacks.custom_accuracy import CustomAccuracy
from callbacks.analytics import DistanceAnalyse

Registry(NLPMlflowLoggerCallback)
Registry(Bert)
Registry(ruBertNLPRunner)