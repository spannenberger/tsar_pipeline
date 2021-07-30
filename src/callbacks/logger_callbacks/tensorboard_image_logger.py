from catalyst.dl import Callback, CallbackOrder
from catalyst.registry import Registry
from catalyst.core.runner import IRunner
import numpy as np
import pandas as pd
import ast
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm
from utils.utils import get_from_dict
from pathlib import Path


@Registry
class TensorboardMulticlassLoggingCallback(Callback):

    def __init__(self, logging_image_number, **kwargs):
        self.logging_image_number = logging_image_number
        super().__init__(CallbackOrder.ExternalExtra)

    def on_experiment_end(self, state: IRunner):
        """В конце эксперимента логаем ошибочные фотографии, раскидывая их в N папок,
        которые соответствуют class_names в нашем конфиге
        """

        df = pd.read_csv(get_from_dict(
            state.hparams, 'stages:stage:callbacks:infer:subm_file'), sep=';')

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
        print('We start logging images to tensorboard... please wait')
        for i in tqdm(range(length)):
            image = ToTensor()(Image.open(f"{path_list[i]}"))
            state.loggers['tensorboard'].loggers['valid'].add_image(
                f"{class_names[target[i]]}/{class_id[i]} - {target[i]} error number {i}.png",
                image)


@Registry
class TensorboardMultilabelLoggingCallback(Callback):
    def __init__(self, logging_image_number, threshold=0.5):
        self.logging_image_number = logging_image_number
        self.threshold = threshold
        super().__init__(CallbackOrder.ExternalExtra)

    def on_experiment_end(self, state: IRunner):
        """В конце эксперимента логаем ошибочные фотографии, раскидывая их в N папок,
        которые соответствуют class_names в нашем конфиге
        """

        df = pd.read_csv(get_from_dict(
            state.hparams, 'stages:stage:callbacks:infer:subm_file'), sep=';')

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
        print('\nWe start logging images to tensorboard... please wait')
        for i in tqdm(range(length)):
            error_ind = np.where(df['class_id'][i] != df['target'][i])[0]
            for ind in tqdm(error_ind):
                image = ToTensor()(Image.open(f"{paths_list[i]}"))
                state.loggers['tensorboard'].loggers['valid'].add_image(
                    f"{class_names[ind]}/{df['class_id'][i][ind]} - {df['target'][i][ind]} error number {i}.png",
                    image)


@Registry
class TensorboardMetricLearningCallback(Callback):

    def __init__(self, logging_incorrect_image_number=5, logging_uncoordinated_image_number=5):
        self.logging_incorrect_image_number = logging_incorrect_image_number
        self.logging_uncoordinated_image_number = logging_uncoordinated_image_number
        super().__init__(CallbackOrder.ExternalExtra)

    def on_experiment_end(self, state: IRunner):
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
            incorrect_image = ToTensor()(Image.open(incorrect_list[i]))
            couple_image = ToTensor()(Image.open(couple_list[i]))
            state.loggers['tensorboard'].loggers['valid'].add_image(
                f'incorrect/{incorrect_list[i].split("/")[2]}/{i}/incorrect.png',
                incorrect_image
            )
            state.loggers['tensorboard'].loggers['valid'].add_image(
                f'incorrect/{incorrect_list[i].split("/")[2]}/{i}/couple.png',
                couple_image
            )
        for i in tqdm(range(uncoordinated_length)):
            uncoordinated_image = ToTensor()(Image.open(uncoordinated_list[i]))
            image_path = Path(uncoordinated_list[i])
            save_path = Path('uncoordinated/') / image_path.parts[2] / image_path.name
            state.loggers['tensorboard'].loggers['valid'].add_image(
                save_path.as_posix(),
                uncoordinated_image
            )
