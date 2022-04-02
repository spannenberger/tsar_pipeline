from re import L
from torch import embedding, nn
import torch

class FeatureExtractionModel(nn.Module):
    def __init__(self, model: nn.Module, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.backbone = model

    def get_embeddings(self, X):
        return self.backbone(X)

    def forward(self, X):
        return self.get_embeddings(X)

class ClassificationModel(FeatureExtractionModel):
    def __init__(self, model: nn.Module, classificator: nn.Module, embedding_size):
        super().__init__(model, embedding_size)
        self.classificator = classificator

    def replace_classificator(self, new_classificator: nn.Module, embedding_size):
        self.classificator = new_classificator

    def forward(self, X):
        embeddings = self.get_embeddings(X)
        return self.classificator(embeddings)


class MetricLearningModel(FeatureExtractionModel):
    pass

class GPTFeatureExtractionModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model
    
    def get_embeddings(self):
        return self.backbone.get_output_embeddings().in_features

    def forward(self, input_ids, attention_mask, labels):
        tmp = self.backbone(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return tmp
class NLPModel(GPTFeatureExtractionModel):
    pass

class SiameseFeatureExtractionModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model
        embeddings = self.get_embeddings()
        self.fc = nn.Linear(embeddings, 1)

    def get_embeddings(self):
        return self.backbone.get_input_embeddings().embedding_dim

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

class SiameseModel(SiameseFeatureExtractionModel):
    pass