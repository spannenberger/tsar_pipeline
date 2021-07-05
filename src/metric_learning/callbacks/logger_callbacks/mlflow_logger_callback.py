from catalyst.dl import Callback, CallbackOrder
from catalyst.registry import Registry
from catalyst.core.runner import IRunner
import mlflow
from tqdm import tqdm

@Registry
class MetricLearningLogger(Callback):

    def __init__(self):
        super().__init__(CallbackOrder.ExternalExtra)


    def on_stage_start(self, state: IRunner):
        mlflow.log_artifact(state.hparams['args']['configs'][0], 'config')


    def on_experiment_end(self, state: IRunner):
        checkpoint_names = ['last', 'best_full']
        print('We start logging convert models... please wait')
        for model in tqdm(checkpoint_names):
            try:
                mlflow.log_artifact(f'logs/logs/torchsript/{model}.pt', 'torchscript_models')
            except FileNotFoundError:
                print(f'No such file {model}.pt, nothing to log...')
            try:
                mlflow.log_artifact(f'logs/logs/onnx/{model}.onnx', 'onnx_models')
            except FileNotFoundError:
                print(f'No such file {model}.onnx, nothing to log...')
        mlflow.pytorch.log_model(state.model, artifact_path=state.hparams['model']['_target_'])
        mlflow.end_run()