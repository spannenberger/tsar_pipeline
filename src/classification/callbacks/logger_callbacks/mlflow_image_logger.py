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
        mlflow.log_artifact(state.hparams['args']['configs'][0], 'config')
        mlflow.log_artifact(
            state.hparams['stages']['stage']['data']['transform_path'], 'config/aug_config')
        try:
            mlflow.log_artifact(state.hparams['stages']['stage']['callbacks']['triton']['conf_path'], 'config/triton')
        except FileNotFoundError:
            print('Сant find triton config, because you disabled this callback')
            print('\n'*3)

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
        print('Start logging images to mlflow... please wait')
        for i in tqdm(range(length)):
            image = Image.open(f"{path_list[i]}")
            mlflow.log_image(
                image,
                f"{class_names[target[i]]}/{class_id[i]} - {target[i]} error number {i}.png")

        if 'quantization' in state.hparams['stages']['stage']['callbacks']:
            mlflow.log_artifact('logs/quantized.pth', 'quantized_model')
        else:
            print('No such file quantized.pth, because quantization callback is disabled')
        
        onnx_checkpoint_names = state.hparams['stages']['stage']['callbacks']['onnx_saver']['checkpoint_names']
        torchsript_checkpoint_names = state.hparams['stages']['stage']['callbacks']['torchscript_saver']['checkpoint_names']

        print('Starting logging convert models... please wait')
        for model in tqdm(torchsript_checkpoint_names):
            try:
                mlflow.log_artifact(f'logs/logs/torchsript/{model}.pt', 'torchscript_models')
            except FileNotFoundError:
                print(f'No such file {model}.pt, nothing to log...')
        
        for model in tqdm(onnx_checkpoint_names):
            try:
                mlflow.log_artifact(f'logs/logs/onnx/{model}.onnx', 'onnx_models')
            except FileNotFoundError:
                print(f'No such file {model}.onnx, nothing to log...')

        if 'prunning' in state.hparams['stages']['stage']['callbacks']:
            mlflow.log_artifact('logs/checkpoints/last.pth', 'prunned_models')
            mlflow.log_artifact('logs/checkpoints/best.pth', 'prunned_models')
        else:
            print('No prunned models to log')

        mlflow.pytorch.log_model(state.model, artifact_path=state.hparams['model']['_target_'])
        mlflow.end_run()

@Registry
class MLFlowMultilabelLoggingCallback(Callback):
    def __init__(self, logging_image_number, threshold=0.5):
        self.logging_image_number = logging_image_number
        self.threshold = threshold
        super().__init__(CallbackOrder.ExternalExtra)

    def on_stage_start(self, state: IRunner):
        """Логаем конфиг эксперимента и аугментации как артефакт в начале стейджа"""

        mlflow.log_artifact(state.hparams['args']['configs'][0], 'config')
        mlflow.log_artifact(
            state.hparams['stages']['stage']['data']['transform_path'], 'config/aug_config')
        try:
            mlflow.log_artifact(state.hparams['stages']['stage']['callbacks']['triton']['conf_path'], 'config/triton')
        except FileNotFoundError:
            print('Сant find triton config, because you disabled this callback')
            print('\n'*3)

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

        print('Start logging images to mlflow... please wait')
        for i in tqdm(range(length)):
            error_ind = np.where(df['class_id'][i] != df['target'][i])[0]
            for ind in tqdm(error_ind):
                    f"{class_names[ind]}/{df['class_id'][i][ind]} - {df['target'][i][ind]} error number {i}.png"

        if 'quantization' in state.hparams['stages']['stage']['callbacks']:
            mlflow.log_artifact('logs/quantized.pth', 'quantized_model')
        else:
            print('No such file quantized.pth, because quantization callback is disabled')

        onnx_checkpoint_names = state.hparams['stages']['stage']['callbacks']['onnx_saver']['checkpoint_names']
        torchsript_checkpoint_names = state.hparams['stages']['stage']['callbacks']['torchscript_saver']['checkpoint_names']

        print('Starting logging convert models... please wait')
        for model in tqdm(torchsript_checkpoint_names):
            try:
                mlflow.log_artifact(f'logs/logs/torchsript/{model}.pt', 'torchscript_models')
            except FileNotFoundError:
                print(f'No such file {model}.pt, nothing to log...')
        
        for model in tqdm(onnx_checkpoint_names):
            try:
                mlflow.log_artifact(f'logs/logs/onnx/{model}.onnx', 'onnx_models')
            except FileNotFoundError:
                print(f'No such file {model}.onnx, nothing to log...')
        
        if 'prunning' in state.hparams['stages']['stage']['callbacks']:
            mlflow.log_artifact('logs/checkpoints/last.pth', 'prunned_models')
            mlflow.log_artifact('logs/checkpoints/best.pth', 'prunned_models')
        else:
            print('No prunned models to log')

        mlflow.pytorch.log_model(state.model, artifact_path=state.hparams['model']['_target_'])
        mlflow.end_run()
