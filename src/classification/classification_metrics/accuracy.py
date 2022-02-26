from typing import Any, Dict
from catalyst.dl import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.registry import Registry
from sklearn.metrics import accuracy_score
from pathlib import Path
from torch.nn import BCELoss, Sigmoid
from torch.nn import CrossEntropyLoss
import torch

from metrics.custom_metric import CustomMetric


class MainClassificationAccuracy(CustomMetric):
    """
    кастомный колбэк
    Записывает в файл preds.csv, в нашем случае,
    картинку(ее путь), то что предсказала наша модель,
    то что должна была предсказать и лосс этой картинки(использовалось для
    просмотра фотографий с большим лоссом)
    """

    def __init__(
        self,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None,
    ):
        super().__init__(False, prefix, suffix)
        self.final = []
        self.ac = -1
        self.ac_score_f = accuracy_score

    def reset(self, num_batches: int, num_samples: int) -> None:
        self.paths = []
        self.targets = []
        self.preds = []
        self.losses = []

    def compute(self) -> Any:
        return super().compute()

    def compute_key_value(self) -> Dict[str, float]:
        cur_ac = self.ac_score_f(torch.Tensor(self.preds), torch.Tensor(self.targets))
        if cur_ac > self.ac:
            self.ac = cur_ac
            self.final = []
            for path, pred, target, loss in zip(
                self.paths, self.preds, self.targets, self.losses
            ):
                self.final.append((path, pred, target, loss.tolist()))
        return {"CustomAccuracy": self.ac}

    def metric_to_string(self):
        subm = ["path;class_id;target;losses"]
        subm += [f"{path};{cls};{tar};{los}" for path, cls, tar, los in self.final]
        metric_dict = {}
        metric_dict["predicted_images"] = "\n".join(subm) + "\n"
        return metric_dict


class MultilabelAccuracy(MainClassificationAccuracy):
    def __init__(
        self,
        threshold,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None
    ):
        super().__init__(compute_on_call, prefix, suffix)
        self.act = Sigmoid()
        self.threshold = threshold
        self.loss_f = BCELoss(reduction="none")

    def update(self, *args, **kwargs) -> None:
        self.paths += kwargs["image_name"]
        targets = kwargs["targets"].detach().cpu()
        logits = kwargs["logits"].detach().cpu().type(torch.DoubleTensor)
        self.targets += targets.tolist()
        logits = self.act(logits)
        self.losses += list(self.loss_f(logits, targets))
        self.preds += kwargs["for_metrics"].tolist()


class MulticlassAccuracy(MainClassificationAccuracy):
    def __init__(
        self,
        compute_on_call: bool = True,
        prefix: str = None,
        suffix: str = None
    ):
        super().__init__(compute_on_call, prefix, suffix)
        self.loss_f = CrossEntropyLoss(reduction="none")

    def update(self, *args, **kwargs) -> None:
        self.paths += kwargs["image_name"]
        targets = kwargs["targets"].detach().cpu()
        logits = kwargs["logits"].detach().cpu()
        self.targets += targets.tolist()
        self.losses += list(self.loss_f(logits, targets))
        self.preds += torch.argmax(logits, dim=1).tolist()
