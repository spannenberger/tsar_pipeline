from sklearn.neighbors import KNeighborsClassifier
from metrics.custom_metric import CustomMetric
from sklearn.metrics import accuracy_score
from typing import Any, Dict
import numpy as np
import torch


class MetricLearningAccuracy(CustomMetric):
    """
    кастомный колбэк
    Записывает в файл preds.csv, в нашем случае,
    картинку(ее путь), то что предсказала наша модель,
    то что должна была предсказать и лосс этой картинки(использовалось для
    просмотра фотографий с большим лоссом)
    """

    def __init__(
        self,
        threshold,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        super().__init__(False, prefix, suffix)
        self.threshold = threshold
        self.accuracy = -1
    
    
    def metric_to_string(self):
        subm = ["incorrect;couple"]
        subm += [f"{incorrect};{couple}" for incorrect, couple in self.final['incorrect']]
        metric_dict = {}
        metric_dict["incorrect"] = "\n".join(subm)+"\n"
        subm = ["uncoordinated"]
        subm += self.final['uncoordinated'].tolist()
        metric_dict["uncoordinated"] = "\n".join(subm)+"\n"
        return metric_dict

    def reset(self, num_batches: int, num_samples: int) -> None:
        self.preds = None
        self.paths = None
        self.targets = None
        self.is_query = None
        self.knn = KNeighborsClassifier(n_neighbors=1)
        

    def compute(self) -> Any:
        return super().compute()
    

    def update(self, *args, **kwargs) -> None:
        paths = kwargs["image_name"]
        targets = kwargs["targets"].detach().cpu()
        logits = kwargs["embeddings"].detach().cpu().type(torch.DoubleTensor)
        is_query = torch.Tensor(kwargs["is_query"].tolist()).bool()
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


    def compute_key_value(self) -> Dict[str, float]:
        X = self.preds[self.is_query == False]
        Y = self.targets[self.is_query == False]
        self.knn.fit(X, Y)
        x = self.preds[self.is_query == True]
        y = self.targets[self.is_query == True]
        predicts = torch.Tensor(self.knn.predict(x))
        current_accuracy = accuracy_score(predicts, y)
        if current_accuracy > self.accuracy:
            self.accuracy = current_accuracy
            distance, idx = self.knn.kneighbors(
                x, n_neighbors=1, return_distance=True
            )
            uncoordinated = distance > self.threshold
            uncoordinated = uncoordinated.reshape(-1)
            incorrect = predicts != y
            incorrect = (incorrect & ~uncoordinated).bool()
            self.final = {"uncoordinated": [], "incorrect": []}
            self.final["uncoordinated"] = self.paths[self.is_query == True][
                uncoordinated
            ]
            incorrect_paths = self.paths[self.is_query == True]
            incorrect_paths = incorrect_paths[incorrect]
            couple_incorrect_paths = []
            for i in idx:
                i = i[0]
                couple_incorrect_paths.append(self.paths[self.is_query == False][i])
            self.final["incorrect"] = [
                [incorrect, couple]
                for incorrect, couple in zip(
                    incorrect_paths, couple_incorrect_paths
                )
            ]
        return {"CustomAccuracy": self.accuracy}