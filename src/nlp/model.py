from transformers import BertForSequenceClassification
from torch import nn
import torch
import torchvision as vision

class Bert(nn.Module):

    def __init__(self, num_classes, model_path, **kwargs):
        self.model_path = model_path
        self.num_classes = num_classes
        super().__init__()
        self.backbone = BertForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels = self.num_classes,
            output_attentions = False,
            output_hidden_states = False
            )

    def forward(self, input_ids, attention_mask, token_type_ids):
        tmp = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        return tmp