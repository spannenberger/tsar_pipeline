from catalyst.callbacks.metric import LoaderMetricCallback
from catalyst import dl

from classification_metrics.accuracy import MulticlassAccuracy, MultilabelAccuracy

class MulticlassCustomAccuracy(dl.ControlFlowCallback):
    def __init__(self, loaders, *args, **kwargs):
        super().__init__(base_callback=MulticlassAccuracyCallback(*args, **kwargs), loaders=loaders)


class MulticlassAccuracyCallback(LoaderMetricCallback):
    def __init__(
        self,
        image_name,
        targets,
        logits,
    ):
        """Init."""
        super().__init__(
            metric=MulticlassAccuracy(),
            input_key=[logits, image_name],
            target_key=[targets],
        )

class MultilabelCustomAccuracy(dl.ControlFlowCallback):
    def __init__(self, loaders, *args, **kwargs):
        super().__init__(base_callback=MultilabelAccuracyCallback(*args, **kwargs), loaders=loaders)


class MultilabelAccuracyCallback(LoaderMetricCallback):
    def __init__(
        self,
        image_name,
        targets,
        logits,
        for_metrics,
        threshold
    ):
        """Init."""
        super().__init__(
            metric=MultilabelAccuracy(
                threshold=threshold
            ),
            input_key=[logits, for_metrics, image_name],
            target_key=[targets],
        )

