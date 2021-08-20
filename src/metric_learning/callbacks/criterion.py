from catalyst.callbacks.criterion import CriterionCallback
from catalyst import dl
from arcface_loss import AngularPenaltySMLoss


class CustomCriterion(CriterionCallback):
    def __init__(self, num_classes, loss_type='arcface', scale=None, margin=None, *args, **kwargs):
        self.loss_type = loss_type
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        super().__init__(*args, **kwargs)

    def on_stage_start(self, runner):
        self.criterion = AngularPenaltySMLoss(
            runner.model.embedding_size, self.num_classes, scale=self.scale, margin=self.margin, loss_type=self.loss_type)


class CustomTrainCriterion(dl.ControlFlowCallback):
    def __init__(self, loaders, *args, **kwargs):
        super().__init__(CustomCriterion(*args, **kwargs), loaders=loaders)
