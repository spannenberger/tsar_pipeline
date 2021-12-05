from transformers import GPT2LMHeadModel
from torch import nn

class gpt(nn.Module):

    def __init__(self, model_name="sberbank-ai/rugpt3small_based_on_gpt2", **kwargs):
        super().__init__()
        self.model_name = model_name
        self.backbone = GPT2LMHeadModel.from_pretrained(self.model_name)


    def forward(self, input_ids, attention_mask, labels):
        tmp = self.backbone(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return tmp
