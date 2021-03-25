from catalyst.core import IRunner
from catalyst.runners import SupervisedConfigRunner
from collections import OrderedDict
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from dataset import CustomDataset
from catalyst.data.transforms import ToTensor
from pprint import pprint
class TTARunner(IRunner): # Кастомный runner нашего эксперимента

    # Реализация ТТА 
    def _run_stage(self, *args,**kwargs):
        try:
            self.tta = self.hparams['stages'][self.stage_key]['data']['tta']
        except Exception as e:
            self.tta = 1
        super()._run_stage(*args,**kwargs)

    def _run_batch(self):
        self._run_event("on_batch_start")
        self.handle_batch(batch=self.batch)

        if self.is_valid_loader and self.tta > 1:
            # resize inputs
            self.batch['image_name'] = [x for x in self.batch['image_name'][::self.tta]]
            self.batch['targets']= torch.stack([x for x in self.batch['targets'][::self.tta]])
            # averaging outputs
            logits = self.batch['logits']
            self.batch['logits'] = torch.stack([logits[i: i + self.tta].mean(dim = 0)/self.tta for i in range(0, len(logits) - self.tta + 1, self.tta)])
        self.batch['for_metrics'] = (self.batch['logits'] > self.hparams['args']['threshold']).type(torch.ByteTensor)
        self._run_event("on_batch_end")

    # Работа с данными, формирование train и valid 
    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()
        data_params = self._stage_config[stage]["data"]
        image_size = data_params.get("image_size")
        if image_size is not None:
            assert len(image_size) == 2
            image_size = tuple(image_size)

        train_dir = Path(data_params["train_dir"])
        train_path = train_dir.joinpath(data_params["train_meta"])
        test_path = train_dir.joinpath(data_params["test_meta"])

        train_images_dir = train_dir.joinpath(data_params["train_image_dir"])
        test_images_dir = train_dir.joinpath(data_params["test_image_dir"])

        train_meta = pd.read_csv(train_path)
        test_meta = pd.read_csv(test_path)
        train_meta['label'] = [i[0] for i in zip(train_meta.iloc[:, 1:].values)]
        test_meta['label'] = [i[0] for i in zip(test_meta.iloc[:, 1:].values)]
        # train_meta["label"] = train_meta["label"].astype(np.int64)

        train_image_paths = [train_images_dir.joinpath(i) for i in train_meta["path"]]
        test_image_paths = [test_images_dir.joinpath(i) for i in test_meta["path"]]

        train_labels = train_meta["label"].tolist()
        test_labels = test_meta["label"].tolist()
        image_paths_train = train_image_paths
        labels_train = train_labels
        image_paths_val = test_image_paths
        labels_val = test_labels

        tta = data_params.get('tta', 1)
        datasets["train"] = {'dataset':CustomDataset(image_paths_train,
                                            labels_train,
                                            transforms_path=data_params['transform_path']
                                            )
                                }
        datasets["valid"] = CustomDataset(image_paths_val,
                                          labels_val,
                                          transforms_path=data_params['transform_path'],
                                          valid = True,
                                          tta = tta)
        return datasets
class TTASupervisedRunner(TTARunner, SupervisedConfigRunner):
    pass
