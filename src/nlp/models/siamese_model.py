import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
import torch.nn.functional as F


class Siamese(nn.Module):

    def __init__(self, model_name="sberbank-ai/rugpt3small_based_on_gpt2", embeddings_number=768):
        super().__init__()
        self.embs = embeddings_number
        self.model_name = model_name 
        self.backbone = GPT2LMHeadModel.from_pretrained(self.model_name).transformer
        self.fc = nn.Linear(self.embs, 1)

    def forward_one(self, x):
        x = self.backbone(**x).last_hidden_state
        x = x.mean(dim=2)
        x = x.view(x.size()[0], -1)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.fc(dis)
        return out