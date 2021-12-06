# flake8: noqa
import torch
from catalyst.runners import ConfigRunner

class YOLOXDetectionRunner(ConfigRunner):
    """Runner for YOLO-X models."""

    def get_model(self, *args, **kwargs):
        return super().get_model(*args, **kwargs)()

    def get_loaders(self, stage: str):
        """Insert into loaders collate_fn.

        Args:
            stage (str): stage name

        Returns:
            ordered dict with torch.utils.data.DataLoader
        """
        # import pdb;pdb.set_trace()
        loaders = super().get_loaders(stage)
        for item in loaders.values():
            if hasattr(item.dataset, "collate_fn"):
                item.collate_fn = item.dataset.collate_fn
        return loaders

    def handle_batch(self, batch):
        """Do a forward pass and compute loss.

        Args:
            batch (Dict[str, Any]): batch of data.
        """

        if self.is_train_loader:
            images = batch["image"]
            targets = torch.cat([batch["labels"].unsqueeze(-1), batch["bboxes"]], -1)
            loss = self.model(images, targets)
            self.batch_metrics["loss"] = loss
        else:
            predictions = self.model(batch["image"])
            self.batch["predicted_tensor"] = predictions