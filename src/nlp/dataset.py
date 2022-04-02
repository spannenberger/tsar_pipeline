from torch.utils.data import Dataset
import torch


class CustomNLPDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = {}
        item["story_1"] = self.df.iloc[idx]["story_1"]
        item["story_2"] = self.df.iloc[idx]["story_2"]
        item["labels"] = self.df.iloc[idx]["label"]
        return item

class NLPDataset(Dataset):
    def __init__(self, input_ids, attention_mask, masked=0.15, labels=True):
        self.input_ids = input_ids
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
        item["attention_mask"] = self.attention_mask[idx]
        if self.labels:
            item["labels"] = self.input_ids[idx]
        return item

