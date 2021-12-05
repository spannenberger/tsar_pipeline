from catalyst.dl import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.registry import Registry
from pathlib import Path

@Registry
class SaveModelWithConfigCallback(Callback):
    """Функция для сохранения модели с конфигом"""

    def __init__(self, save_path: str):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(CallbackOrder.ExternalExtra)

    def on_experiment_end(self, state: IRunner):
        state.model.backbone.save_pretrained(self.save_path, save_config=True)
