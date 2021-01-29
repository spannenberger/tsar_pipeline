from catalyst.dl import registry, Callback, CallbackOrder, State
import pandas as pd


@registry.Callback
class InerCallback(Callback):

    def __init__(self, subm_file, fold_index = 0, num_folds = 0):
        self.fold_index = fold_index
        self.num_folds = num_folds
        super().__init__(CallbackOrder.Internal)
        self.subm_file = subm_file
        self.preds = []

    def on_batch_end(self, state: State):
        if state.loader_name == 'valid' and state.epoch == state.num_epochs:
            paths = state.input["image_name"]
            targets = state.input['targets']
            preds = state.output["logits"].detach().cpu().numpy()
            preds = preds.argmax(axis=1)
            for path, pred, target in zip(paths, preds, targets):
                self.preds.append((path, pred, target))
    

    def on_experiment_end(self, _):
        
        if not hasattr(self,'num_folds'):
            subm = ["path,class_id,target"]
            subm += [f"{path},{cls},{tar}" for path, cls, tar in self.preds]
            with open(self.subm_file, 'w') as file:
                file.write("\n".join(subm)+"\n")
        else:
            if self.fold_index == 0:
                subm = ["path,class_id,target"]
                subm += [f"{path},{cls},{tar}" for path, cls, tar in self.preds]
                with open(self.subm_file, 'w') as file:
                    file.write("\n".join(subm)+"\n")
            else:
                subm = [f"{path},{cls},{tar}" for path, cls, tar in self.preds]
                with open(self.subm_file, 'a') as file:
                    file.write("\n".join(subm)[1:]+"\n")