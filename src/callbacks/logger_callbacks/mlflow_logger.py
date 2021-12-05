from catalyst.loggers.mlflow import MLflowLogger, EXCLUDE_PARAMS, STAGE_PARAMS, EXPERIMENT_PARAMS, _mlflow_log_dict
from typing import Dict, Any
import mlflow

ALLOWED_PARAMS_EXPERIMENT = {
    'model._target_': 'model_name',
    'args.seed': 'seed',
}

ALLOWED_PARAMS_STAGE = {
    'optimizer._target_': 'optimizer',
    'optimizer.lr': 'lr',
    'loaders.batch_size': 'batch_size',
    'loaders.num_workers': 'num_workers',
    'num_epochs': 'num_epochs',
    'main_metric': 'main_metric',
    'minimize_metric': 'minimize_metric'
}


class CustomMLflowLogger(MLflowLogger):
    def __init__(self, class_names=None, *args, **kwargs):
        self.class_names = class_names
        super().__init__(*args, **kwargs)

    def log_hparams(
        self,
        hparams: Dict,
        scope: str = None,
        # experiment info
        run_key: str = None,
        stage_key: str = None,
    ) -> None:
        stages = set(hparams.get("stages", {})) - set(STAGE_PARAMS) - set(EXCLUDE_PARAMS)
        self._multistage = len(stages) > 1

        if scope == "experiment":
            if self._multistage:
                mlflow.set_tag("mlflow.runName", run_key)

        if scope == "stage":
            if self._multistage:
                mlflow.start_run(run_name=stage_key, nested=True)

            scope_params = hparams.get("stages", {}).get(run_key, {})
            _mlflow_log_dict(scope_params, log_type="param")

            stage_params = hparams.get("stages", {}).get(stage_key, {})
            new_stage_params = {}
            for param in ALLOWED_PARAMS_STAGE:
                params = stage_params
                for i in param.split('.'):
                    try:
                        params = params[i]
                    except KeyError as e:
                        raise KeyError(
                            f"config.yaml has not key '{param}' key '{e.args[0]}' is wrong.")
                new_stage_params[ALLOWED_PARAMS_STAGE[param]] = params
            _mlflow_log_dict(new_stage_params, log_type="param")

            exp_params = hparams
            new_exp_params = {}
            for param in ALLOWED_PARAMS_EXPERIMENT:
                params = exp_params
                for i in param.split('.'):
                    try:
                        params = params[i]
                    except KeyError as e:
                        raise KeyError(
                            f"config.yaml has not key '{param}' key '{e.args[0]}' is wrong.")
                new_exp_params[ALLOWED_PARAMS_EXPERIMENT[param]] = params
            _mlflow_log_dict(new_exp_params, log_type="param")

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        scope: str = None,
        # experiment info
        run_key: str = None,
        global_epoch_step: int = 0,
        global_batch_step: int = 0,
        global_sample_step: int = 0,
        # stage info
        stage_key: str = None,
        stage_epoch_len: int = 0,
        stage_epoch_step: int = 0,
        stage_batch_step: int = 0,
        stage_sample_step: int = 0,
        # loader info
        loader_key: str = None,
        loader_batch_len: int = 0,
        loader_sample_len: int = 0,
        loader_batch_step: int = 0,
        loader_sample_step: int = 0,
    ) -> None:
        """Logs batch and epoch metrics to MLflow."""
        if scope == "epoch":
            for loader_key, per_loader_metrics in metrics.items():
                # тут временное решение(костыль), но чтобы фиксить названия
                # классов нужно лезть в метрики
                change_keys = {}

                for item in per_loader_metrics:
                    if "class" not in item:
                        continue
                    metric_name, class_name = item.split("/")
                    new_item = f"{metric_name}/{self.class_names[int(class_name.split('_')[1])]}"
                    change_keys[new_item] = item
                for i in change_keys:
                    per_loader_metrics[i] = per_loader_metrics.pop(change_keys[i])

                self._log_metrics(
                    metrics=per_loader_metrics,
                    step=global_epoch_step,
                    loader_key=loader_key,
                    suffix="",
                )
