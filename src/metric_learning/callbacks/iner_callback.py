from catalyst.dl import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.registry import Registry
from sklearn.metrics import accuracy_score
from pathlib import Path
from torch.nn import BCELoss, Sigmoid
from torch.nn import CrossEntropyLoss
from sklearn.neighbors import KNeighborsClassifier
import torch
import numpy as np


@Registry
class MLInerCallback(Callback):
    """
    InerCallback - кастомный колбэк
    Записывает в файл preds.csv, в нашем случае,
    картинку(ее путь), то что предсказала наша модель,
    то что должна была предсказать и лосс этой картинки(использовалось для
    просмотра фотографий с большим лоссом)
    """

    def __init__(self,
                 subm_file,
                 **kwargs):
        super().__init__(CallbackOrder.Internal)
        self.subm_file = Path(subm_file)
        self.subm_file.parent.mkdir(parents=True, exist_ok=True)

    def on_loader_start(self, _):
        self.preds = None
        self.paths = None
        self.targets = None
        self.is_query = None
        self.knn = KNeighborsClassifier(n_neighbors=1)

    def on_loader_end(self, state: IRunner):
        print("\n"*5)
        print(self.preds)
        print(self.is_query)
        # neigh.fit(X, y)
        # cur_ac = self.ac_score_f(torch.Tensor(self.preds), torch.Tensor(self.targets))
        # if state.is_valid_loader:
        #     if cur_ac > self.ac:
        #         self.ac = cur_ac
        #         self.final = []
        #         for path, pred, target, loss in zip(self.paths, self.preds, self.targets, self.losses):
        #             self.final.append((path, pred, target, loss.tolist()))

    # def on_experiment_end(self, _):
    #     subm = ["path;class_id;target;losses"]
    #     subm += [f"{path};{cls};{tar};{los}" for path, cls, tar, los in self.final]
    #     with self.subm_file.open(mode='w') as file:
    #         file.write("\n".join(subm)+"\n")

    def on_batch_end(self, state: IRunner):
        if state.is_valid_loader:
            paths = state.batch["image_name"]
            targets = state.batch['targets'].detach().cpu()
            logits = state.batch["embeddings"].detach().cpu().type(torch.DoubleTensor)
            is_query = torch.Tensor(state.batch['is_query'].tolist()).bool()
            if self.preds is None:
                self.paths = np.array(paths)
                self.preds = logits
                self.is_query = is_query
                self.targets = targets
            else:
                self.paths = np.concatenate([self.paths, paths])
                self.preds = torch.cat([self.preds, logits], dim=0)
                self.is_query = torch.cat([self.is_query, is_query], dim=0)
                self.targets = torch.cat([self.targets, targets], dim=0)
