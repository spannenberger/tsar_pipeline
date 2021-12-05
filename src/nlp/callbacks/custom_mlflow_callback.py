from catalyst.dl import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.registry import Registry
import mlflow

@Registry
class CustomMLFlowCallback(Callback):

    def __init__(self):
        super().__init__(CallbackOrder.ExternalExtra)

    def on_experiment_end(self, state: IRunner):
        mlflow.log_artifact(state.hparams["stages"]["stage"]["callbacks"]["analyze"]["similarity_file_path"], "distance analyze")
        mlflow.log_artifact(state.hparams["stages"]["stage"]["callbacks"]["analyze"]["cluster_file_path"], "clusters")
