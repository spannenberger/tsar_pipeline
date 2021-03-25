from catalyst.dl import Callback, CallbackOrder
from catalyst.registry import Registry
from catalyst.core.runner import IRunner
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
import pandas as pd
import torch
import ast
from PIL import Image
import numpy as np
from pprint import pprint


@Registry
class MLFlowloggingCallback(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.ExternalExtra)

    def on_stage_start(self, state: IRunner):
        # Логаем конфиг эксперимента и аугментации как артефакт в начале стейджа
        mlflow.log_artifact(state.hparams['stages']['stage']['data']['transform_path'], 'config')
        mlflow.log_artifact(state.hparams['args']['configs'][0],'config')

    def on_experiment_end(self, state: IRunner):
        df = pd.read_csv('crossval_log/preds.csv', sep=',')

        path_list = [i for i in df[df['class_id']!=df['target']]['path']]
        class_names = state.hparams['class_names']
        for i in range(len(path_list)):
            image = Image.open(f"{path_list[i]}")
            mlflow.log_image(image, f"image{i}.png")

        mlflow.log_artifact('logs/checkpoints/best.pth', 'model')
        mlflow.end_run()


if __name__ == "__main__":
    a = MLFlowloggingCallback(experiment_name = 'test', is_locally=True, env_path='', model_path='', classes_list='')
