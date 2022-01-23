from sklearn.neighbors import KNeighborsClassifier
from catalyst.dl import Callback, CallbackOrder
from sklearn.metrics import accuracy_score
from catalyst.core.runner import IRunner
from catalyst.registry import Registry
from pathlib import Path
import numpy as np
import torch


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
                 incorrect_file,
                 uncoordinated_file,
                 threshold,
                 **kwargs):
        super().__init__(CallbackOrder.Internal)
        self.incorrect_file = Path(incorrect_file).absolute()
        self.uncoordinated_file = Path(uncoordinated_file).absolute()
        self.incorrect_file.parent.mkdir(parents=True, exist_ok=True)
        self.uncoordinated_file.parent.mkdir(parents=True, exist_ok=True)
        self.threshold = threshold
        self.accuracy = -1

    def on_loader_start(self, _):
        self.preds = None
        self.paths = None
        self.targets = None
        self.is_query = None
        self.knn = KNeighborsClassifier(n_neighbors=1)

    def on_loader_end(self, state: IRunner):
        if state.is_valid_loader:
            X = self.preds[self.is_query == False]
            Y = self.targets[self.is_query == False]
            self.knn.fit(X, Y)
            x = self.preds[self.is_query == True]
            y = self.targets[self.is_query == True]
            predicts = torch.Tensor(self.knn.predict(x))
            current_accuracy = accuracy_score(predicts, y)
            if current_accuracy > self.accuracy:
                self.accuracy = current_accuracy
                distance, idx = self.knn.kneighbors(x, n_neighbors=1, return_distance=True)
                uncoordinated = distance > self.threshold
                uncoordinated = uncoordinated.reshape(-1)
                incorrect = predicts != y
                incorrect = (incorrect & ~uncoordinated).bool()
                self.final = {'uncoordinated': [], 'incorrect': []}
                self.final['uncoordinated'] = self.paths[self.is_query == True][uncoordinated]
                incorrect_paths = self.paths[self.is_query == True]
                incorrect_paths = incorrect_paths[incorrect]
                couple_incorrect_paths = []
                for i in idx:
                    i = i[0]
                    couple_incorrect_paths.append(self.paths[self.is_query == False][i])
                self.final['incorrect'] = [[incorrect, couple]
                                           for incorrect, couple in zip(incorrect_paths, couple_incorrect_paths)]

    def on_experiment_end(self, _):
        subm = ["incorrect;couple"]
        subm += [f"{incorrect};{couple}" for incorrect, couple in self.final['incorrect']]
        with self.incorrect_file.open(mode='w') as file:
            file.write("\n".join(subm)+"\n")
        subm = ["uncoordinated"]
        subm += self.final['uncoordinated'].tolist()
        with self.uncoordinated_file.open(mode='w') as file:
            file.write("\n".join(subm)+"\n")

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
