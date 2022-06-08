from catalyst.dl import Callback, CallbackOrder
from catalyst.registry import Registry
from catalyst.core.runner import IRunner
import mlflow
import pandas as pd
import ast
from PIL import Image
import numpy as np
from tqdm import tqdm
from catalyst.core.callback import Callback, CallbackWrapper
from catalyst.callbacks.metric import LoaderMetricCallback
from metrics.custom_metric import CustomMetric
from utils.utils import get_from_dict
from pathlib import Path

def _get_original_callback(callback: Callback) -> Callback:
    """Docs."""
    while isinstance(callback, CallbackWrapper):
        callback = callback.callback
    return callback
class MainMLFlowLoggerCallback(Callback):

    def __init__(self):
        super().__init__(CallbackOrder.ExternalExtra)

    def on_stage_start(self, state: IRunner):
        """Логаем конфиг эксперимента и аугментации как артефакт в начале стейджа"""

        callbacks = [_get_original_callback(x) for x in state.callbacks.values()]
        all_metrics = [x.metric for x in callbacks if issubclass(type(x), LoaderMetricCallback)]
        include_custom_metric = max([issubclass(type(x), CustomMetric) for x in all_metrics])
        if not include_custom_metric:
            raise Exception("This callback depends on CustomMetric. Turn on CustomMetric or turn off this callback.")

        mlflow.log_artifact(get_from_dict(state.hparams, 'args:configs')[0], 'config')
        mlflow.log_artifact(
            get_from_dict(state.hparams, 'stages:stage:data:transform_path'), 'config/aug_config')

    def on_experiment_end(self, state: IRunner):
        callbacks_dict = get_from_dict(state.hparams, 'stages:stage:callbacks')
        if 'quantization' in callbacks_dict:
            mlflow.log_artifact('logs/quantized.pth', 'quantized_model')
        else:
            print('\nNo such file quantized.pth, because quantization callback is disabled')
        onnx_checkpoint_names = get_from_dict(
            callbacks_dict, 'onnx_saver:checkpoint_names', default=[])
        torchsript_checkpoint_names = get_from_dict(
            callbacks_dict, 'torchscript_saver:checkpoint_names', default=[])

        print('\nStarting logging convert models... please wait')

        if len(torchsript_checkpoint_names) > 0:
            for model in tqdm(torchsript_checkpoint_names, desc="Torchsript"):
                try:
                    path = Path(state.logdir) / get_from_dict(callbacks_dict,
                                                              'torchscript_saver:out_dir') / f'{model}.pt'
                    mlflow.log_artifact(str(path), 'torchscript_models')
                except FileNotFoundError:
                    print(f'\nNo such file {model}.pt, nothing to log...')
        else:
            print("Torchsript convert callback is disabled\n")

        if len(onnx_checkpoint_names) > 0:
            for model in tqdm(onnx_checkpoint_names,desc="Onnx"):
                try:
                    path = Path(state.logdir) / get_from_dict(callbacks_dict,
                                                              'onnx_saver:out_dir') / f'{model}.onnx'
                    mlflow.log_artifact(str(path), 'onnx_models')
                except FileNotFoundError:
                    print(f'\nNo such file {model}.onnx, nothing to log...\n')
        else:
            print("Onnx convert callback is disabled\n")

        if 'prunning' in callbacks_dict:
            path = get_from_dict(callbacks_dict, 'saver:logdir')
            mlflow.log_artifact(f'{path}/last.pth', 'prunned_models')
            mlflow.log_artifact(f'{path}/best.pth', 'prunned_models')
        else:
            print('\nNo prunned models to log\n')

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

        df = pd.read_csv(get_from_dict(state.hparams, 'stages:stage:callbacks:save_metrics:required_metrics:iner:predicted_images'), sep=';')
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

        for i in tqdm(range(length), desc="Logging images to mlflow:"):
            image = Image.open(f"{path_list[i]}")
            image_path = f"{class_names[target[i]]}/{class_id[i]} - {target[i]} error number {i}.png"
            mlflow.log_image(
                image,
                image_path
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

        df = pd.read_csv(get_from_dict(state.hparams, 'stages:stage:callbacks:save_metrics:required_metrics:iner:predicted_images'), sep=';')

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


        for i in tqdm(range(length), desc="Logging images to mlflow:"):
            error_ind = np.where(df['class_id'][i] != df['target'][i])[0]
            for ind in error_ind:
                image = Image.open(f"{paths_list[i]}")
                image_path = f"{class_names[ind]}/{df['class_id'][i][ind]} - {df['target'][i][ind]} error number {i}.png"

                mlflow.log_image(
                    image,
                    image_path
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

        incorrect_df = pd.read_csv(
            get_from_dict(state.hparams, 'stages:stage:callbacks:save_metrics:required_metrics:custom_accuracy:incorrect'), sep=';')
        uncoordinated_df = pd.read_csv(
            get_from_dict(state.hparams, 'stages:stage:callbacks:save_metrics:required_metrics:custom_accuracy:uncoordinated'), sep=';')

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
        for i in tqdm(range(incorrect_length), desc="Logging incorrect images to mlflow"):
            incorrect_image = Image.open(incorrect_list[i])
            couple_image = Image.open(couple_list[i])
            image_path = Path(incorrect_list[i])
            save_path = Path('incorrect/')/Path(incorrect_list[i]).parts[-2]/image_path.name
            mlflow.log_image(
                incorrect_image,
                save_path/'incorrect.png'
            )
            mlflow.log_image(
                couple_image,
                save_path/'couple.png'
            )
        for i in tqdm(range(uncoordinated_length), desc="Logging uncoordinated images to mlflow"):
            uncoordinated_image = Image.open(uncoordinated_list[i])
            image_path = Path(uncoordinated_list[i])
            save_path = Path('uncoordinated/') / Path(image_path.parts[-2]) / image_path.name
            mlflow.log_image(uncoordinated_image, save_path.as_posix())


        super().on_experiment_end(state)

@Registry
class MLFlowNLPLoggingCallback(Callback):

    def __init__(self):
        super().__init__(CallbackOrder.ExternalExtra)

    
    def on_experiment_end(self, state: IRunner):
        callbacks_dict = get_from_dict(state.hparams, 'stages:stage:callbacks')
        if "analyze" in callbacks_dict:
            similar_path = get_from_dict(callbacks_dict, 'analyze:similarity_file_path')
            cluster_path = get_from_dict(callbacks_dict, 'analyze:cluster_file_path')
            mlflow.log_artifact(similar_path, 'distance analyze')
            mlflow.log_artifact(cluster_path, 'clusters')