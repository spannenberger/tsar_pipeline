from catalyst.core import IRunner
from catalyst.runners import SupervisedConfigRunner
import torch

from dataset_fabric import ClassificationDatasetCreator


class MulticlassRunner(IRunner):
    """Кастомный runner нашего эксперимента"""

    def get_datasets(self, stage: str, **kwargs):
        """Работа с данными, формирование train и valid"""

        data_params = self._stage_config[stage]["data"]
        data_params["task_mode"] = "Multiclass"
        datasets = ClassificationDatasetCreator.create_datasets(data_params, **kwargs)
        return datasets


class MulticlassSupervisedRunner(MulticlassRunner, SupervisedConfigRunner):
    pass


class MultilabelRunner(IRunner):
    """Кастомный runner нашего эксперимента"""

    def _run_batch(self):
        self._run_event("on_batch_start")
        self.handle_batch(batch=self.batch)

        self.batch['for_metrics'] = (
            self.batch['logits'] > self.hparams['args']['threshold']).type(torch.ByteTensor)
        self._run_event("on_batch_end")

    def get_datasets(self, stage: str, **kwargs):
        """Работа с данными, формирование train и valid"""

        data_params = self._stage_config[stage]["data"]
        data_params["task_mode"] = "Multilabel"
        datasets = ClassificationDatasetCreator.create_datasets(data_params, **kwargs)
        return datasets

class MultilabelSupervisedRunner(MultilabelRunner, SupervisedConfigRunner):
    pass
