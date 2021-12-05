from catalyst.core import IRunner
from catalyst.runners import SupervisedConfigRunner
from collections import OrderedDict
from pathlib import Path
import pandas as pd
import numpy as np
from dataset import CustomNLPDataset, NLPDataset
import torch
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split
import json
from sklearn.model_selection import train_test_split


class NLPRunner(IRunner):
    """Кастомный runner нашего эксперимента"""

    def handle_batch(self, batch) -> None:
        logits = self.model(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        # import pdb; pdb.set_trace()
        self.batch_metrics['loss'] = logits.loss
        self.batch['logits'] = logits

    def get_datasets(self, stage: str, **kwargs):
        """Работа с данными, формирование train и valid"""
        datasets = OrderedDict()

        with open(self._stage_config[stage]["data"]["text"], 'r', encoding='utf-8') as f:
            shuffled_data = f.readlines()
        shuffled_data = list(set(shuffled_data))

        self.tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2")
        self.tokenizer.add_special_tokens({'pad_token': '<pad>'})

        train_split, val_split = train_test_split(shuffled_data)

        train_data = self.tokenizer(list(train_split), padding=True, truncation=True, max_length=100, return_tensors='pt')
        val_data = self.tokenizer(list(val_split), padding=True, truncation=True, max_length=100, return_tensors='pt')

        datasets["train"] = {'dataset': NLPDataset(**train_data)}
        datasets["valid"] = NLPDataset(**val_data)

        return datasets


class NLPSupervisedRunner(NLPRunner, SupervisedConfigRunner):
    pass


class MulticlassSiameseRunner(IRunner):
    """Кастомный runner нашего эксперимента"""

    def handle_batch(self, batch) -> None:
        logits = self.model(batch['story_1'], batch['story_2'])
        self.batch["labels"] = torch.squeeze(self.batch["labels"]).view(-1, 1).type(torch.FloatTensor)
        # self.batch_metrics['loss'] = logits.loss
        self.batch['logits'] = logits

    def get_datasets(self, stage: str, **kwargs):
        """Работа с данными, формирование train и valid"""
        datasets = OrderedDict()

        df = pd.read_csv(self._stage_config[stage]["data"]["text"])

        self.tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
        self.tokenizer.add_special_tokens({'pad_token': '<pad>'})

        train_split, val_split = train_test_split(df)

        train_split["story_1"] = train_split["story_1"].apply(lambda x : self.tokenizer(x, padding="max_length", truncation=True, max_length=100, return_tensors='pt'))
        train_split["story_2"] = train_split["story_2"].apply(lambda x : self.tokenizer(x, padding="max_length", truncation=True, max_length=100, return_tensors='pt'))

        val_split["story_1"] = val_split["story_1"].apply(lambda x : self.tokenizer(x, padding="max_length", truncation=True, max_length=100, return_tensors='pt'))
        val_split["story_2"] = val_split["story_2"].apply(lambda x : self.tokenizer(x, padding="max_length", truncation=True, max_length=100, return_tensors='pt'))
        
        train_split=train_split.reset_index(drop=True)
        val_split=val_split.reset_index(drop=True)


        datasets["train"] = {'dataset': CustomNLPDataset(train_split)}
        datasets["valid"] = CustomNLPDataset(val_split)

        return datasets


class SiameseSupervisedRunner(MulticlassSiameseRunner, SupervisedConfigRunner):
    pass
