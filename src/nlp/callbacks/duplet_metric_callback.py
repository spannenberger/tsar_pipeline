from catalyst.dl import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.registry import Registry
from dataset import NLPDataset
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.spatial.distance import cosine
import torch
import mlflow

@Registry
class DupletMetricCallback(Callback):
    def __init__(self, duplets_file: str, threshold: float = 0.3):
        super().__init__(CallbackOrder.ExternalExtra)
        self.threshold = threshold
        self.duplets = pd.read_csv(duplets_file)


    def on_experiment_end(self, state: IRunner):

        if state.hparams['model']['_target_'] == 'gpt':
            model = state.model.backbone.transformer
        elif state.hparams['model']['_target_'] == 'Siamese':
            model = state.model.backbone
        tokenizer = state.tokenizer
        data = pd.concat((self.duplets["story_1"], self.duplets["story_2"]))
        tokenized_data = tokenizer(list(data), padding=True, truncation=True, max_length=100, return_tensors='pt')
        dataset = NLPDataset(**tokenized_data ,labels=False)
        loader = DataLoader(dataset, shuffle=False, batch_size=1)
        outputs_final = []
        print("calculation duplet meteric")
        with torch.no_grad():
            for batch in tqdm(loader):
                batch = state.engine.sync_device(tensor_or_module=batch)
                outputs = model(**batch)
                outputs_final.append(outputs.last_hidden_state.mean(axis=1).detach().cpu())
        duplets = torch.squeeze(torch.stack(outputs_final))
        duplets = duplets.view(2, -1, duplets.shape[1]).permute(1,0,2)
        acc = []
        for idx, (story_1, story_2) in enumerate(duplets):
            dist = cosine(story_1,story_2)
            acc.append(self.duplets["label"][idx]==(dist<self.threshold))
        final = (sum(acc)/len(acc))
        mlflow.log_metric("duplets_metric", final)
