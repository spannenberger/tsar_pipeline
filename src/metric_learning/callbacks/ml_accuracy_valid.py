from catalyst.callbacks.metric import LoaderMetricCallback
from ml_metrics.ml_accuracy import MetricLearningAccuracy
from catalyst import dl


class CustomAccuracy(dl.ControlFlowCallback):
    def __init__(self, loaders, *args, **kwargs):
        super().__init__(base_callback=MetricLearningAccuracyCallback(*args, **kwargs), loaders=loaders)


class MetricLearningAccuracyCallback(LoaderMetricCallback):
    def __init__(
        self,
        threshold,
        embeddings_key,
        is_query_key,
        image_name,
        labels_key
    ):
        """Init."""
        super().__init__(
            metric=MetricLearningAccuracy(
                threshold=threshold
            ),
            input_key=[embeddings_key, is_query_key, image_name],
            target_key=[labels_key],
        )

