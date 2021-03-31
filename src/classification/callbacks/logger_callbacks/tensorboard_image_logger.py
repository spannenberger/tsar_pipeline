from catalyst.dl import Callback, CallbackOrder
from catalyst.registry import Registry
from catalyst.core.runner import IRunner
import numpy as np
import pandas as pd
import ast
from PIL import Image
from torchvision.transforms import ToTensor
from tqdm import tqdm


@Registry
class TensorboardMulticlassLoggingCallback(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.ExternalExtra)

    def on_experiment_end(self, state: IRunner):
        # В конце эксперимента логаем ошибочные фотографии, раскидывая их в N папок, которые соответствуют class_names в нашем конфиге
        df = pd.read_csv('crossval_log/preds.csv', sep=';')

        path_list = [i for i in df[df['class_id'] != df['target']]['path']]
        class_names = state.hparams['class_names']
        for i in tqdm(range(len(path_list))):
            image = ToTensor()(Image.open(f"{path_list[i]}"))
            state.loggers['tensorboard'].loggers['valid'].add_image(f"image{i}.png", image)


@Registry
class TensorboardMultilabelLoggingCallback(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.ExternalExtra)

    def on_experiment_end(self, state: IRunner):
        # В конце эксперимента логаем ошибочные фотографии, раскидывая их в N папок, которые соответствуют class_names в нашем конфиге
        df = pd.read_csv('crossval_log/preds.csv', sep=';')

        df[['class_id', 'target', 'losses']] = df[['class_id', 'target', 'losses']].apply(
            lambda x: x.apply(ast.literal_eval))
        df['class_id'] = df['class_id'].apply(lambda x: [1.0 if i > 0.5 else 0.0 for i in x])

        length = len(df[df['class_id'] != df['target']])
        paths_list = df[df['class_id'] != df['target']]['path']

        df['class_id'] = df['class_id'].apply(lambda x: np.array([1.0 if i > 0.5 else 0.0 for i in x]))
        df['class_id'] = df['class_id'].apply(lambda x: np.array(x))

        class_names = state.hparams['class_names']
        for i in tqdm(range(length)):
            error_ind = np.where(df['class_id'][i] != df['target'][i])[0]
            for ind in tqdm(error_ind):
                image = ToTensor()(Image.open(f"{paths_list[i]}"))
                state.loggers['tensorboard'].loggers['valid'].add_image(f"{class_names[ind][1:]}/image{i}.png", image)
