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
        paths = state.input["image_name"]
        targets = state.input['targets']
        preds = state.output["logits"].detach().cpu().numpy()
        preds = preds.argmax(axis=1)
        for path, pred, target in zip(paths, preds, targets):
            self.preds.append((path, pred, target))
    

    def on_loader_end(self, _):
        subm = ["path,class_id,target"]
        subm += [f"{path},{cls},{tar}" for path, cls, tar in self.preds]
        if hasattr(self,'num_folds'):
            with open(self.subm_file, 'w') as file:
                file.write("\n".join(subm))
        else:
            if self.fold_index == 0:
                with open(self.subm_file, 'w') as file:
                    file.write("\n".join(subm))
            else:
                with open(self.subm_file, 'a') as file:
                    file.write("\n".join(subm)[1:])