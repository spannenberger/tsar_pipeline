from catalyst.dl import Callback, CallbackOrder
from torchvision.transforms import ToTensor
from catalyst.core.runner import IRunner
from catalyst.registry import Registry
from utils.utils import get_from_dict
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import ast
from catalyst.core.misc import _get_original_callback
from catalyst.callbacks.metric import LoaderMetricCallback
from metrics.custom_metric import CustomMetric

class TensorboardLoggingCallback(Callback):

    def on_stage_start(self, state: IRunner):
        callbacks = [_get_original_callback(x) for x in state.callbacks.values()]
        all_metrics = [x.metric for x in callbacks if issubclass(type(x), LoaderMetricCallback)]
        include_custom_metric = max([issubclass(type(x), CustomMetric) for x in all_metrics])
        if not include_custom_metric:
            raise Exception("This callback depends on CustomMetric. Turn on CustomMetric or turn off this callback.")


@Registry
class TensorboardMulticlassLoggingCallback(TensorboardLoggingCallback):

    def __init__(self, logging_image_number, **kwargs):
        self.logging_image_number = logging_image_number
        super().__init__(CallbackOrder.ExternalExtra)

    def on_experiment_end(self, state: IRunner):
        """В конце эксперимента логаем ошибочные фотографии, раскидывая их в N папок,
        которые соответствуют class_names в нашем конфиге
        """
        df = pd.read_csv(get_from_dict(
            state.hparams, 'stages:stage:callbacks:save_metrics:required_metrics:iner:predicted_images'), sep=';')

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

        for i in tqdm(range(length), desc="Logging images to tensorboard:"):
            image = ToTensor()(Image.open(f"{path_list[i]}"))
            image_path = f"{class_names[target[i]]}/{class_id[i]} - {target[i]} error number {i}.png"

            state.loggers['tensorboard'].loggers['valid'].add_image(
                image_path,
                image)


@Registry
class TensorboardMultilabelLoggingCallback(TensorboardLoggingCallback):
    def __init__(self, logging_image_number, threshold=0.5):
        self.logging_image_number = logging_image_number
        self.threshold = threshold
        super().__init__(CallbackOrder.ExternalExtra)

    def on_experiment_end(self, state: IRunner):
        """В конце эксперимента логаем ошибочные фотографии, раскидывая их в N папок,
        которые соответствуют class_names в нашем конфиге
        """
        df = pd.read_csv(get_from_dict(
            state.hparams, 'stages:stage:callbacks:save_metrics:required_metrics:iner:predicted_images'), sep=';')

        df[['class_id', 'target', 'losses']] = df[['class_id', 'target', 'losses']].apply(
            lambda x: x.apply(ast.literal_eval))

        df['class_id'] = df['class_id'].apply(
            lambda x: [1.0 if i > 0.5 else 0.0 for i in x])

        paths_list = df[df['class_id'] != df['target']]['path']
        if(len(df[df['class_id'] != df['target']]) <= self.logging_image_number):
            length = len(df[df['class_id'] != df['target']])
        else:
            length = self.logging_image_number

        df['class_id'] = df['class_id'].apply(
            lambda x: np.array([1.0 if i > self.threshold else 0.0 for i in x]))

        df['class_id'] = df['class_id'].apply(
            lambda x: np.array(x))

        try:
            class_names = state.hparams['class_names']
        except KeyError:
            class_names = [x for x in range(
                state.hparams['model']['num_classes'])]

        for i in tqdm(range(length), desc="Logging images to tensorboard:"):
            error_ind = np.where(df['class_id'][i] != df['target'][i])[0]
            for ind in error_ind:
                image = ToTensor()(Image.open(f"{paths_list[i]}"))
                image_path = f"{class_names[ind]}/{df['class_id'][i][ind]} - {df['target'][i][ind]} error number {i}.png"

                state.loggers['tensorboard'].loggers['valid'].add_image(
                    image_path,
                    image)


@Registry
class TensorboardMetricLearningCallback(TensorboardLoggingCallback):

    def __init__(self, logging_incorrect_image_number=5, logging_uncoordinated_image_number=5):
        self.logging_incorrect_image_number = logging_incorrect_image_number
        self.logging_uncoordinated_image_number = logging_uncoordinated_image_number
        super().__init__(CallbackOrder.ExternalExtra)

    def on_experiment_end(self, state: IRunner):
        incorrect_df = pd.read_csv(get_from_dict(
            state.hparams, 'stages:stage:callbacks:save_metrics:required_metrics:custom_accuracy:incorrect'), sep=';')
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
        for i in tqdm(range(incorrect_length), desc="Logging incorrect images to tensorboard"):
            incorrect_image = ToTensor()(Image.open(incorrect_list[i]))
            couple_image = ToTensor()(Image.open(couple_list[i]))
            incorrect_path = f"incorrect/{Path(incorrect_list[i]).parts[-2]}/{i}/incorrect.png"
            state.loggers['tensorboard'].loggers['valid'].add_image(
                incorrect_path,
                incorrect_image
            )
            couple_path = f"incorrect/{Path(incorrect_list[i]).parts[-2]}/{i}/couple.png"
            state.loggers['tensorboard'].loggers['valid'].add_image(
                couple_path,
                couple_image
            )
        for i in tqdm(range(uncoordinated_length), desc="Logging uncoordinated images to tensorboard"):
            uncoordinated_image = ToTensor()(Image.open(uncoordinated_list[i]))
            image_path = Path(uncoordinated_list[i])
            save_path = Path('uncoordinated/').absolute() / image_path.parts[-2] / image_path.name
            state.loggers['tensorboard'].loggers['valid'].add_image(
                save_path.as_posix(),
                uncoordinated_image
            )
