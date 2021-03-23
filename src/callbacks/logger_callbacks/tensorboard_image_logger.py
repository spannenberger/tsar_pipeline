from catalyst.dl import Callback, CallbackOrder
from catalyst.registry import Registry
from catalyst.core.runner import IRunner
# from pprint import pprint
import numpy as np
import pandas as pd
import torch
import ast
from PIL import Image
from torchvision.transforms import ToTensor

@Registry
class TensorboardImageCustomLogger(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.ExternalExtra)

    def on_experiment_end(self, state: IRunner):
        df = pd.read_csv('crossval_log/preds.csv', sep=',')

        path_list = [i for i in df[df['class_id']!=df['target']]['path']]
        class_names = state.hparams['class_names']
        for i in range(len(path_list)):
            image = ToTensor()(Image.open(f"{path_list[i]}"))

            state.loggers['tensorboard'].loggers['valid'].add_image(f"image{i}.png", image)
