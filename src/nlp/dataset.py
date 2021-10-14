import numpy as np
import torch
from torch.utils.data import Dataset


class NLPDataset(Dataset):
    """
    Работа с данными для nlp задач
    """

    def __init__(self, input_ids, token_type_ids, attention_mask, labels, masked=0.15):
        # import pdb; pdb.set_trace()
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.masked = masked
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {}
        input_ids = self.input_ids[idx].detach().clone()

        mask = torch.rand(input_ids.shape) < self.masked # marks for masked part
        mask *= input_ids != 101 # remove marks for special tokens
        mask *= input_ids != 102
        input_ids[mask] = 103 # add mask tokens
        
        item["input_ids"] = input_ids
        item["token_type_ids"] = self.token_type_ids[idx]
        item["attention_mask"] = self.attention_mask[idx]
        item["targets"] = self.labels[idx]
        return item