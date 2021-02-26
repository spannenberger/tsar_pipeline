from catalyst.dl import registry, Callback, CallbackOrder, State
from sklearn.metrics import accuracy_score
import pandas as pd
from pathlib import Path
from sklearn.metrics import log_loss
from torch.nn import CrossEntropyLoss


@registry.Callback
class InerCallback(Callback):

    def __init__(self, subm_file, fold_index = 0, num_folds = 0):
        self.fold_index = fold_index
        self.num_folds = num_folds
        super().__init__(CallbackOrder.Internal)
        self.subm_file = Path(subm_file)
        self.subm_file.parent.mkdir(parents=True, exist_ok=True)
        self.final = []
        self.ac = -1
        self.loss_f = CrossEntropyLoss(reduce=False)


    def on_batch_end(self, state: State):
        if state.is_valid_loader:
            self.paths += state.input["image_name"]
            self.targets += state.input['targets'].tolist()
            preds = state.output["logits"].detach().cpu().numpy()
            self.preds += list(preds.argmax(axis=1))
            self.losses += list(self.loss_f(state.output["logits"].detach().cpu(),state.input['targets'].detach().cpu()))
            
    def on_loader_start(self, _):
        self.paths = []
        self.targets = []
        self.preds = []
        self.losses = []

    def on_loader_end(self, state: State):
        if state.is_valid_loader:
            cur_ac = accuracy_score(self.preds, self.targets)
            if cur_ac > self.ac:
                self.ac = cur_ac
                self.final = []
                for path, pred, target, loss in zip(self.paths, self.preds, self.targets, self.losses):
                    self.final.append((path, pred, target, loss))


    def on_experiment_end(self, _):
        if self.num_folds == 0:
            subm = ["path,class_id,target,losses"]
            subm += [f"{path},{cls},{tar},{los}" for path, cls, tar, los in self.final]
            with self.subm_file.open(mode = 'w') as file:
                file.write("\n".join(subm)+"\n")
        else:
            if self.fold_index == 0:
                subm = ["path,class_id,target,losses"]
                subm += [f"{path},{cls},{tar},{los}" for path, cls, tar, los in self.final]
                with self.subm_file.open(mode = 'w') as file:
                    file.write("\n".join(subm)+"\n")
            else:
                subm = [f"{path},{cls},{tar},{los}" for path, cls, tar, los in self.final]
                with self.subm_file.open(mode = 'a') as file:
                    file.write("\n".join(subm)+"\n")