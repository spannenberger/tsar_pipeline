from catalyst.dl import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.registry import Registry
from sklearn.metrics import accuracy_score
import pandas as pd
from pathlib import Path
from sklearn.metrics import log_loss
from torch.nn import BCELoss, Sigmoid
from catalyst.metrics._accuracy import MultilabelAccuracyMetric
from torch.nn import CrossEntropyLoss
import torch

@Registry
class InerCallback(Callback):
    """
    InerCallback - кастомный колбэк
    Записывает в файл preds.csv, в нашем случае,
    картинку(ее путь), то что предсказала наша модель,
    то что должна была предсказать и лосс этой картинки(использовалось для
    просмотра фотографий с большим лоссом)
    """
    def __init__(self,
                subm_file,
                loss='None',
                accuracy='None',
                activation='None',
                fold_index = 0,
                num_folds = 0,
                **kwargs):
        self.func_args = kwargs
        self.args = {
        'BCEloss':BCELoss(reduction='none'),
        'CrossEntropyLoss':CrossEntropyLoss(reduction='none'),
        'Sigmoid':Sigmoid(),
        'Multilabel':MultilabelAccuracyMetric(),
        'Multiclass':accuracy_score,
        'Argmax': torch.argmax
        }

        self.fold_index = fold_index
        self.num_folds = num_folds
        super().__init__(CallbackOrder.Internal)
        self.subm_file = Path(subm_file)
        self.subm_file.parent.mkdir(parents=True, exist_ok=True)
        self.final = []
        self.ac = -1
        self.loss_f = self.args[loss]
        self.act = self.args[activation] if activation in self.args else None
        self.ac_score_f =  self.args[accuracy]


    def on_batch_end(self, state: IRunner):
        if state.is_valid_loader:
            self.paths += state.batch["image_name"]
            targets = state.batch['targets'].detach().cpu()
            logits = state.batch["logits"].detach().cpu()
            if self.act != None:
                if 'activaton_args' in self.func_args:
                    logits = self.act(logits,**self.func_args['activaton_args'])
                else:
                    logits = self.act(logits)
            logits = logits.type(torch.DoubleTensor)
            self.losses += list(self.loss_f(logits, targets))
            self.targets += targets.tolist()
            self.preds += logits.tolist()

    def on_loader_start(self, _):
        self.paths = []
        self.targets = []
        self.preds = []
        self.losses = []

    def on_loader_end(self, state: IRunner):
        if state.is_valid_loader:
            cur_ac = self.ac_score_f(torch.Tensor(self.preds), torch.Tensor(self.targets))
            if cur_ac[0] > self.ac:
                self.ac = cur_ac[0]
                self.final = []
                for path, pred, target, loss in zip(self.paths, self.preds, self.targets, self.losses):
                    self.final.append((path, pred, target, loss.tolist()))


    def on_experiment_end(self, _):
        if self.num_folds == 0:
            subm = ["path;class_id;target;losses"]
            subm += [f"{path};{cls};{tar};{los}" for path, cls, tar, los in self.final]
            with self.subm_file.open(mode = 'w') as file:
                file.write("\n".join(subm)+"\n")
        else:
            if self.fold_index == 0:
                subm = ["path;class_id;target;losses"]
                subm += [f"{path};{cls};{tar};{los}" for path, cls, tar, los in self.final]
                with self.subm_file.open(mode = 'w') as file:
                    file.write("\n".join(subm)+"\n")
            else:
                subm = [f"{path};{cls};{tar};{los}" for path, cls, tar, los in self.final]
                with self.subm_file.open(mode = 'a') as file:
                    file.write("\n".join(subm)+"\n")
