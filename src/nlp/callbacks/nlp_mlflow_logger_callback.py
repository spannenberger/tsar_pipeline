import numpy as np
from catalyst.dl import Callback, CallbackOrder
from catalyst.registry import Registry
from catalyst.core.runner import IRunner
import pandas as pd
import mlflow
from utils.utils import get_from_dict

class NLPMlflowLoggerCallback(Callback):

    def __init__(self):
        super().__init__(CallbackOrder.ExternalExtra)

    def on_stage_start(self, state: IRunner):
        mlflow.log_artifact(get_from_dict(state.hparams, 'args:configs')[0], 'config')
