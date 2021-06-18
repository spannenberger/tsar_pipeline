from catalyst import dl


class CustomCMC(dl.ControlFlowCallback):
    def __init__(self, loaders, *args, **kwargs):
        super().__init__(base_callback=dl.CMCScoreCallback(*args, **kwargs), loaders=loaders)
