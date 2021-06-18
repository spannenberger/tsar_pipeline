from catalyst.callbacks.criterion import CriterionCallback
from catalyst import dl
from .archface_loss import AngularPenaltySMLoss


class CustomCriterion(CriterionCallback):
    def __init__(self, embeding_size, num_classes, *args, **kwargs):
        self.embeding_size = embeding_size
        self.num_classes = num_classes
        super().__init__(*args, **kwargs)

    def on_stage_start(self, runner):
        self.criterion = AngularPenaltySMLoss(
            self.embeding_size, self.num_classes, loss_type='arcface')


class CustomTrainCriterion(dl.ControlFlowCallback):
    def __init__(self, loaders, *args, **kwargs):
        super().__init__(CustomCriterion(*args, **kwargs), loaders=loaders)
