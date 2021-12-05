from catalyst import dl
from .custom_cmc_score import CustomCMCScoreCallback


class CustomCMC(dl.ControlFlowCallback):
    def __init__(self, loaders, *args, **kwargs):
        super().__init__(base_callback=CustomCMCScoreCallback(*args, **kwargs), loaders=loaders)