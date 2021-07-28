from catalyst.dl import Callback, CallbackOrder
from catalyst.registry import Registry
from catalyst.core.runner import IRunner
import mlflow
import pandas as pd
import ast
from PIL import Image
import numpy as np
from tqdm import tqdm
from utils.utils import get_from_dict


class MainMLFlowLoggerCallback(Callback):

    def __init__(self):
        super().__init__(CallbackOrder.ExternalExtra)

    def on_stage_start(self, state: IRunner):
        """Логаем конфиг эксперимента и аугментации как артефакт в начале стейджа"""
        mlflow.log_artifact(get_from_dict(state.hparams, 'args:configs')[0], 'config')
        mlflow.log_artifact(
            get_from_dict(state.hparams, 'stages:stage:data:transform_path'), 'config/aug_config')
        try:
            mlflow.log_artifact(get_from_dict(state.hparams, 'stages:stage:callbacks:triton:conf_path'),
                                'config/triton')
        except FileNotFoundError:
            print('Сant find triton config, because you disabled this callback')
            print('\n'*3)

    def on_experiment_end(self, state: IRunner):

        if 'quantization' in get_from_dict(state.hparams, 'stages:stage:callbacks'):
            mlflow.log_artifact('logs/quantized.pth', 'quantized_model')
        else:
            print('No such file quantized.pth, because quantization callback is disabled')

        onnx_checkpoint_names = get_from_dict(
            state.hparams, 'stages:stage:callbacks:onnx_saver:checkpoint_names')
        torchsript_checkpoint_names = get_from_dict(
            state.hparams, 'stages:stage:callbacks:torchscript_saver:checkpoint_names')

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

        if 'prunning' in get_from_dict(state.hparams, 'stages:stage:callbacks'):
            mlflow.log_artifact('logs/checkpoints/last.pth', 'prunned_models')
            mlflow.log_artifact('logs/checkpoints/best.pth', 'prunned_models')
        else:
            print('No prunned models to log')

        mlflow.pytorch.log_model(state.model, artifact_path=get_from_dict(
            state.hparams, 'model:_target_'))
        mlflow.end_run()


@Registry
class MLFlowMulticlassLoggingCallback(MainMLFlowLoggerCallback):

    def __init__(self, logging_image_number, **kwargs):
        self.logging_image_number = logging_image_number
        super().__init__()

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
            class_names = get_from_dict(state.hparams, 'class_names')
        except KeyError:
            class_names = [x for x in range(
                get_from_dict(state.hparams, 'model:num_classes'))]
        print('Start logging images to mlflow... please wait')
        for i in tqdm(range(length)):
            image = Image.open(f"{path_list[i]}")
            mlflow.log_image(
                image,
                f"{class_names[target[i]]}/{class_id[i]} - {target[i]} error number {i}.png"
            )

        super().on_experiment_end(state)


@Registry
class MLFlowMultilabelLoggingCallback(MainMLFlowLoggerCallback):

    def __init__(self, logging_image_number, threshold=0.5):
        self.logging_image_number = logging_image_number
        self.threshold = threshold
        super().__init__()

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
            class_names = get_from_dict(state.hparams, 'class_names')
        except KeyError:
            class_names = [x for x in range(
                get_from_dict(state.hparams, 'model:num_classes'))]

        print('Start logging images to mlflow... please wait')
        for i in tqdm(range(length)):
            error_ind = np.where(df['class_id'][i] != df['target'][i])[0]
            for ind in tqdm(error_ind):
                image = Image.open(f"{paths_list[i]}")
                mlflow.log_image(
                    image,
                    f"{class_names[ind]}/{df['class_id'][i][ind]} - {df['target'][i][ind]} error number {i}.png"
                )

        super().on_experiment_end(state)


@Registry
class MLFlowMetricLearningCallback(MainMLFlowLoggerCallback):
    def __init__(self, logging_incorrect_image_number=5, logging_uncoordinated_image_number=5):
        self.logging_incorrect_image_number = logging_incorrect_image_number
        self.logging_uncoordinated_image_number = logging_uncoordinated_image_number
        super().__init__()

    def on_experiment_end(self, state: IRunner):
        """В конце эксперимента логаем в одну папку ошибочные фотографии и фотографии к которым модель отнесла ошибочную,
        в другую папку логаем фотографии которые не прошли по расстоянию
        """
        incorrect_df = pd.read_csv(get_from_dict(
            state.hparams, 'stages:stage:callbacks:iner:incorrect_file'), sep=';')
        uncoordinated_df = pd.read_csv(
            get_from_dict(state.hparams, 'stages:stage:callbacks:iner:uncoordinated_file'), sep=';')

        incorrect_list = [i for i in incorrect_df['incorrect']]
        couple_list = [i for i in incorrect_df['couple']]
        uncoordinated_list = [i for i in uncoordinated_df['uncoordinated']]

        if self.logging_incorrect_image_number >= len(incorrect_list):
            incorrect_length = len(incorrect_list)
        else:
            incorrect_length = self.logging_incorrect_image_number
        if self.logging_uncoordinated_image_number >= len(uncoordinated_list):
            uncoordinated_length = len(uncoordinated_list)
        else:
            uncoordinated_length = self.logging_uncoordinated_image_number
        for i in tqdm(range(incorrect_length)):
            incorrect_image = Image.open(incorrect_list[i])
            couple_image = Image.open(couple_list[i])
            mlflow.log_image(
                incorrect_image,
                f'incorrect/{incorrect_list[i].split("/")[2]}/{i}/incorrect.png'
            )
            mlflow.log_image(
                couple_image,
                f'incorrect/{incorrect_list[i].split("/")[2]}/{i}/couple.png'
            )
        for i in tqdm(range(uncoordinated_length)):
            uncoordinated_image = Image.open(uncoordinated_list[i])
            mlflow.log_image(
                uncoordinated_image,
                f'uncoordinated/{uncoordinated_list[i].split("/")[2]}/{uncoordinated_list[i].split("/")[3]}.png'
            )
        super().on_experiment_end(state)
