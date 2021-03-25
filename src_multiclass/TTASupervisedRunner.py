from catalyst.core import IRunner
from catalyst.runners import SupervisedConfigRunner
from collections import OrderedDict
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from dataset import CustomDataset
from catalyst.data.transforms import ToTensor
from pprint import pprint
class TTARunner(IRunner):

    # def _run_experiment(self, *args,**kwargs):

    #     try:
    #         self.tta = self.experiment.hparams['stages']['data_params']['tta']
    #     except Exception as e:
    #         self.tta = 1

    #     super()._run_experiment(*args,**kwargs)


    # def _run_batch(self):
        
    #     self.input = self._handle_device(batch=self.input)
    #     self._run_event("on_batch_start")
    #     self._handle_batch(batch=self.input)
    #     #print(dir(self.experiment.get_loaders(stage=self.stage)['train']))
    #     if self.is_valid_loader:
    #         # resize inputs
    #         self.input['image_name'] = [x for x in self.input['image_name'][::self.tta]]
    #         self.input['targets']= torch.stack([x for x in self.input['targets'][::self.tta]])
    #         # averaging outputs
    #         logits = self.output['logits']
    #         self.output['logits'] = torch.stack([logits[i: i + self.tta].mean(dim = 0) for i in range(0, len(logits) - self.tta + 1, self.tta)])
    #     self._run_event("on_batch_end")
    def get_datasets(self, stage: str, **kwargs):
            datasets = OrderedDict()
            data_params = self._stage_config[stage]["data"]
            image_size = data_params.get("image_size")
            if image_size is not None:
                assert len(image_size) == 2
                image_size = tuple(image_size)

            train_dir = Path(data_params["train_dir"])
            metadata_path = train_dir.joinpath(data_params["train_meta"])
            images_dir = train_dir.joinpath(data_params["train_image_dir"])

            train_meta = pd.read_csv(metadata_path)
            train_meta["label"] = train_meta["label"].astype(np.int64)

            image_paths = [images_dir.joinpath(i) for i in train_meta["image_id"]]
            labels = train_meta["label"].tolist()
            tta = data_params.get('tta', 1)

            image_paths_train, image_paths_val, \
            labels_train, labels_val = train_test_split(image_paths, labels,
                                                        stratify=labels,
                                                        test_size=data_params["valid_size"])

            datasets["train"] = {'dataset':CustomDataset(image_paths_train,
                                                labels_train,
                                                transforms_path=data_params['transform_path'])
                                    }
            datasets["valid"] = CustomDataset(image_paths_val,
                                            labels_val,
                                            transforms_path=data_params['transform_path'],
                                            valid = True,
                                            tta = tta)


            return datasets
class TTASupervisedRunner(TTARunner, SupervisedConfigRunner):
    pass
