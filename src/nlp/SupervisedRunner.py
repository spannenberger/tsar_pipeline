from sklearn.model_selection import train_test_split
from catalyst.runners import SupervisedConfigRunner
from dataset import CustomNLPDataset, NLPDataset
from transformers import GPT2Tokenizer
from collections import OrderedDict
from catalyst.core import IRunner
import pandas as pd
import torch

def create_tokenizer(tokenizer_name: str):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    return tokenizer

class NLPRunner(IRunner):
    """Кастомный runner нашего эксперимента"""

    def handle_batch(self, batch) -> None:
        logits = self.model(batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        self.batch_metrics['loss'] = logits.loss
        self.batch['logits'] = logits

    def get_datasets(self, stage: str, **kwargs):
        """Работа с данными, формирование train и valid"""
        datasets = OrderedDict()

        with open(self._stage_config[stage]["data"]["text"], 'r', encoding='utf-8') as f:
            shuffled_data = f.readlines()
        shuffled_data = list(set(shuffled_data))

        tokenizer = create_tokenizer(self._config["model"]["model_name"])

        train_split, val_split = train_test_split(shuffled_data)

        datasets["train"] = {'dataset': NLPDataset(train_split, tokenizer)}
        datasets["valid"] = NLPDataset(val_split, tokenizer)

        return datasets


class NLPSupervisedRunner(NLPRunner, SupervisedConfigRunner):
    pass


class MulticlassSiameseRunner(IRunner):
    """Кастомный runner нашего эксперимента"""

    def handle_batch(self, batch) -> None:
        logits = self.model(batch['story_1'], batch['story_2'])
        self.batch["labels"] = torch.squeeze(self.batch["labels"]).view(-1, 1).type(torch.FloatTensor)
        self.batch['logits'] = logits

    def get_datasets(self, stage: str, **kwargs):
        """Работа с данными, формирование train и valid"""
        datasets = OrderedDict()

        df = pd.read_csv(self._stage_config[stage]["data"]["text"])
        tokenizer = create_tokenizer(self._config["model"]["model_name"])

        train_split, val_split = train_test_split(df)

        train_split["story_1"] = train_split["story_1"].apply(lambda x : tokenizer(x, padding="max_length", truncation=True, max_length=100, return_tensors='pt'))
        train_split["story_2"] = train_split["story_2"].apply(lambda x : tokenizer(x, padding="max_length", truncation=True, max_length=100, return_tensors='pt'))

        val_split["story_1"] = val_split["story_1"].apply(lambda x : tokenizer(x, padding="max_length", truncation=True, max_length=100, return_tensors='pt'))
        val_split["story_2"] = val_split["story_2"].apply(lambda x : tokenizer(x, padding="max_length", truncation=True, max_length=100, return_tensors='pt'))
        
        train_split=train_split.reset_index(drop=True)
        val_split=val_split.reset_index(drop=True)


        datasets["train"] = {'dataset': CustomNLPDataset(train_split)}
        datasets["valid"] = CustomNLPDataset(val_split)

        return datasets


class SiameseSupervisedRunner(MulticlassSiameseRunner, SupervisedConfigRunner):
    pass
