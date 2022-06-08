from catalyst.runners import SupervisedConfigRunner
from catalyst.core import IRunner

from dataset_fabric import MetricLearningDatasetCreator


class MertricLearningRunner(IRunner):
    """Кастомный runner нашего эксперимента"""

    def handle_batch(self, batch) -> None:
        if self.is_train_loader:
            images, targets = batch["features"].float(), batch["targets"].long()
            features = self.model(images)
            self.batch = {
                "embeddings": features,
                "targets": targets,
            }
        else:
            images, targets, is_query = (
                batch["features"].float(),
                batch["targets"].long(),
                batch["is_query"].bool(),
            )
            features = self.model(images)
            self.batch = {
                "embeddings": features,
                "targets": targets,
                "is_query": is_query,
                "image_name": batch["image_name"],
            }

    def get_datasets(self, stage: str, **kwargs):
        """Работа с данными, формирование train и valid"""
        data_params = self._stage_config[stage]["data"]
        data_params["is_check"] = self.hparams["args"].get("check", False)
        data_params["task_mode"] = self._config["model"]["mode"]
        datasets = MetricLearningDatasetCreator.create_datasets(data_params, **kwargs)
        return datasets


class MertricLearningSupervisedRunner(MertricLearningRunner, SupervisedConfigRunner):
    pass
