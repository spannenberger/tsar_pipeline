from catalyst.dl import registry, Callback, CallbackOrder, State
from catalyst.metrics import f1_score
from catalyst.callbacks import LoaderMetricCallback
from catalyst.metrics.functional import (
    wrap_class_metric2dict,
)
@registry.Callback
class CustomF1Score(LoaderMetricCallback):
    def __init__(
    self,
    input_key: str = "targets",
    output_key: str = "logits",
    prefix: str = "f1",
    per_class: bool = False,
    **kwargs,
        ):
        metric_fn = wrap_class_metric2dict(
            f1_score, class_args=None
        )
        super().__init__(
            prefix=prefix,
            metric_fn=metric_fn,
            input_key=input_key,
            output_key=output_key,
            **kwargs,
        )


if __name__ == "__main__":
    a = CustomF1Score()
