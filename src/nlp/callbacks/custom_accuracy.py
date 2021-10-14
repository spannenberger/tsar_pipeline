from catalyst.dl import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.registry import Registry
import torch
import pandas as pd
from transformers import BertTokenizer
from scipy.spatial.distance import cosine
from tqdm import tqdm


@Registry
class CustomAccuracy(Callback):
    def __init__(self):
        super().__init__(CallbackOrder.ExternalExtra)

    def on_batch_end(self, state: IRunner):
        if state.is_valid_loader:
            pass
