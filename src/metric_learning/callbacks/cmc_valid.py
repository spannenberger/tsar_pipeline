from typing import Iterable
from catalyst import dl

from catalyst.callbacks.metric import LoaderMetricCallback
from ml_metrics.custom_cmc_score import CustomCMCMetric


class CustomCMC(dl.ControlFlowCallback):
    def __init__(self, loaders, *args, **kwargs):
        super().__init__(base_callback=CustomCMCScoreCallback(*args, **kwargs), loaders=loaders)

class CustomCMCScoreCallback(LoaderMetricCallback):
    def __init__(
        self,
        embeddings_key: str,
        labels_key: str,
        is_query_key: str,
        topk_args: Iterable[int] = None,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=CustomCMCMetric(
                embeddings_key=embeddings_key,
                labels_key=labels_key,
                is_query_key=is_query_key,
                topk_args=topk_args,
                prefix=prefix,
                suffix=suffix,
            ),
            input_key=[embeddings_key, is_query_key],
            target_key=[labels_key],
        )
