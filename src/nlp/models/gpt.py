from transformers import GPT2LMHeadModel
from torch import nn
import torch
import torchvision as vision

class gpt(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = GPT2LMHeadModel.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")


    def forward(self, input_ids, attention_mask, labels):
        tmp = self.backbone(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return tmp
