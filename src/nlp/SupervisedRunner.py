#! /usr/bin/env python
# -*- coding: utf-8 -*-

from catalyst.core import IRunner
from catalyst.runners import SupervisedConfigRunner
from collections import OrderedDict
import pandas as pd
import numpy as np
from dataset import NLPDataset
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForMaskedLM, BertModel, BertTokenizer, AutoModel

class NLPRunner(IRunner):
    """
    Кастомный runner нашего эксперимента
    """
    def handle_batch(self, batch) -> None:
        logits = self.model(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'])[0]
        self.batch['logits'] = logits
    
    def get_datasets(self, stage: str, **kwargs):
        """Работ с данныи, токенизация, формирование loader"""
        datasets = OrderedDict()
        data_params = self._stage_config[stage]["data"]
        dataset_path = data_params["dataset_path"]

        df = pd.read_csv(dataset_path)

        train_pct_index = int(0.83 * len(df))
        train_df = df[:train_pct_index].sample(frac = 1)
        test_df = df[train_pct_index:].sample(frac = 1)
        X_train, X_test = train_df.sentence, test_df.sentence
        y_train, y_test = train_df.labels, test_df.labels

        tokenizer = BertTokenizer.from_pretrained('sberbank-ai/ruBert-large')

        inputs_train = tokenizer(X_train.tolist(), padding=True, truncation=True, max_length=150, return_tensors="pt")
        labels_train = torch.tensor(y_train.tolist())

        inputs_test = tokenizer(X_test.tolist(), padding=True, truncation=True, max_length=150, return_tensors="pt")
        labels_test = torch.tensor(y_test.tolist())

        datasets["train"] = {'dataset': NLPDataset(**inputs_train, labels=labels_train)}
        datasets["valid"] = NLPDataset(**inputs_test, labels=labels_test)

        return datasets

class ruBertNLPRunner(NLPRunner, SupervisedConfigRunner):
    pass