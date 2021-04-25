from catalyst.dl import Callback, CallbackOrder
from catalyst.registry import Registry
from catalyst.core.runner import IRunner
import mlflow
import pandas as pd
import ast
from PIL import Image
import numpy as np
from tqdm import tqdm
import subprocess
from pprint import pprint


@Registry
class MLFlowMulticlassLoggingCallback(Callback):
    def __init__(self, logging_image_number, **kwargs):
        self.logging_image_number = logging_image_number
        super().__init__(CallbackOrder.ExternalExtra)

    def on_stage_start(self, state: IRunner):
        """Логаем конфиг эксперимента и аугментации как артефакт в начале стейджа"""

        mlflow.log_artifact(
            state.hparams['stages']['stage']['data']['transform_path'], 'config')
        mlflow.log_artifact(state.hparams['args']['configs'][0], 'config')

    def on_experiment_end(self, state: IRunner):
        """В конце эксперимента логаем ошибочные фотографии, раскидывая их в N папок,
        которые соответствуют class_names в нашем конфиге
        """

        df = pd.read_csv('crossval_log/preds.csv', sep=';')
        path_list = [i for i in df[df['class_id'] != df['target']]['path']]
        if(len(df[df['class_id'] != df['target']]) <= self.logging_image_number):
            length = len(df[df['class_id'] != df['target']])
        else:
            length = self.logging_image_number
        class_id = [i for i in df[df['class_id'] != df['target']]['class_id']]
        target = [i for i in df[df['class_id'] != df['target']]['target']]
        try:
            class_names = state.hparams['class_names']
        except KeyError:
            class_names = [x for x in range(
                state.hparams['model']['num_classes'])]
        for i in tqdm(range(length)):
            image = Image.open(f"{path_list[i]}")
            mlflow.log_image(
                image,
                f"{class_names[target[i]]}/{class_id[i]} - {target[i]} error number {i}.png")

        mlflow.log_artifact('logs/logs/torchsript/best_full.pt', 'torchscript_models')
        mlflow.log_artifact('logs/logs/torchsript/best.pt', 'torchscript_models')
        
        mlflow.log_artifact('logs/logs/onnx/best_full.onnx', 'onnx_models')
        mlflow.log_artifact('logs/logs/onnx/best.onnx', 'onnx_models')

        mlflow.pytorch.log_model(state.model, artifact_path=state.hparams['model']['_target_'])
        mlflow.end_run()


@Registry
class MLFlowMultilabelLoggingCallback(Callback):
    def __init__(self, logging_image_number, threshold=0.5):
        self.logging_image_number = logging_image_number
        self.threshold = threshold
        super().__init__(CallbackOrder.ExternalExtra)

    def on_experiment_end(self, state: IRunner):
        """В конце эксперимента логаем ошибочные фотографии, раскидывая их в N папок,
        которые соответствуют class_names в нашем конфиге
        """

        df = pd.read_csv('crossval_log/preds.csv', sep=';')

        df[['class_id', 'target', 'losses']] = df[['class_id', 'target', 'losses']].apply(
            lambda x: x.apply(ast.literal_eval))

        df['class_id'] = df['class_id'].apply(
            lambda x: [1.0 if i > self.threshold else 0.0 for i in x])
        if(len(df[df['class_id'] != df['target']]) <= self.logging_image_number):
            length = len(df[df['class_id'] != df['target']])
        else:
            length = self.logging_image_number

        paths_list = df[df['class_id'] != df['target']]['path']

        df['class_id'] = df['class_id'].apply(
            lambda x: np.array(x))
        try:
            class_names = state.hparams['class_names']
        except KeyError:
            class_names = [x for x in range(
                state.hparams['model']['num_classes'])]
        for i in tqdm(range(length)):
            error_ind = np.where(df['class_id'][i] != df['target'][i])[0]
            for ind in tqdm(error_ind):
                image = Image.open(f"{paths_list[i]}")
                mlflow.log_image(
                    image,
                    f"{class_names[ind]}/{df['class_id'][i][ind]} - {df['target'][i][ind]} error number {i}.png")

        mlflow.log_artifact('logs/logs/torchsript/best_full.pt', 'torchscript_models')
        mlflow.log_artifact('logs/logs/torchsript/best.pt', 'torchscript_models')

        mlflow.log_artifact('logs/logs/onnx/best_full.onnx', 'onnx_models')
        mlflow.log_artifact('logs/logs/onnx/best.onnx', 'onnx_models')

        mlflow.pytorch.log_model(state.model, artifact_path=state.hparams['model']['_target_'])
        mlflow.end_run()
