from catalyst.dl import Callback, CallbackOrder
from catalyst.registry import Registry
from catalyst.core.runner import IRunner
import mlflow
import pandas as pd
import ast
from PIL import Image
import numpy as np
from tqdm import tqdm


@Registry
class MLFlowMulticlassLoggingCallback(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.ExternalExtra)

    def on_stage_start(self, state: IRunner):
        # Логаем конфиг эксперимента и аугментации как артефакт в начале стейджа
        mlflow.log_artifact(state.hparams['stages']['stage']['data']['transform_path'], 'config')
        mlflow.log_artifact(state.hparams['args']['configs'][0], 'config')

    def on_experiment_end(self, state: IRunner):
        # В конце эксперимента логаем ошибочные фотографии, раскидывая их в N папок, которые соответствуют class_names в нашем конфиге
        df = pd.read_csv('crossval_log/preds.csv', sep=';')

        path_list = [i for i in df[df['class_id'] != df['target']]['path']]
        class_id = [i for i in df[df['class_id'] != df['target']]['class_id']]
        target = [i for i in df[df['class_id'] != df['target']]['target']]

        class_names = state.hparams['class_names']
        for i in tqdm(range(len(path_list))):
            image = Image.open(f"{path_list[i]}")
            mlflow.log_image(
                image, 
                f"{class_names[target[i]]}/{class_id[i]} - {target[i]} error number {i}.png")

        mlflow.log_artifact('logs/checkpoints/best.pth', 'model')
        mlflow.end_run()


@Registry
class MLFlowMultilabelLoggingCallback(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.ExternalExtra)

    def on_stage_start(self, state: IRunner):
        # Логаем конфиг эксперимента и аугментации как артефакт в начале стейджа
        mlflow.log_artifact(state.hparams['stages']['stage']['data']['transform_path'], 'config')
        mlflow.log_artifact(state.hparams['args']['configs'][0], 'config')

    def on_experiment_end(self, state: IRunner):
        # В конце эксперимента логаем ошибочные фотографии, раскидывая их в N папок, которые соответствуют class_names в нашем конфиге
        df = pd.read_csv('crossval_log/preds.csv', sep=';')

        df[['class_id', 'target', 'losses']] = df[['class_id', 'target', 'losses']].apply(
            lambda x: x.apply(ast.literal_eval))

        df['class_id'] = df['class_id'].apply(
            lambda x: [1.0 if i > 0.5 else 0.0 for i in x])

        length = len(df[df['class_id'] != df['target']])
        paths_list = df[df['class_id'] != df['target']]['path']

        df['class_id'] = df['class_id'].apply(
            lambda x: np.array([1.0 if i > 0.5 else 0.0 for i in x]))

        df['class_id'] = df['class_id'].apply(
            lambda x: np.array(x))
            
        class_names = state.hparams['class_names']
        for i in tqdm(range(length)):
            error_ind = np.where(df['class_id'][i] != df['target'][i])[0]
            for ind in tqdm(error_ind):
                image = Image.open(f"{paths_list[i]}")
                mlflow.log_image(
                    image,
                    f"{class_names[ind][1:]}/{df['class_id'][i][ind]} - {df['target'][i][ind]} error number {i}.png")

        mlflow.log_artifact('logs/checkpoints/best.pth', 'model')
        mlflow.end_run()
