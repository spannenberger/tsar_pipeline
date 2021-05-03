from catalyst.callbacks.criterion import CriterionCallback
from catalyst.contrib import nn
from catalyst import data, dl


class CustomCriterion(CriterionCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        sampler_inbatch = data.HardTripletsSampler(norm_required=False)
        self.criterion = nn.TripletMarginLossWithSampler(
            margin=0.5, sampler_inbatch=sampler_inbatch)

    def on_stage_start(self, runner):
        print("lel")


class CustomTrainCriterion(dl.ControlFlowCallback):
    def __init__(self, loaders, *args, **kwargs):
        super().__init__(CustomCriterion(*args, **kwargs), loaders=loaders)
