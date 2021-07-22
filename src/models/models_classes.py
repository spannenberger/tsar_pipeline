from torch import nn


class EmbeddingModel(nn.Module):
    def __init__(self, model: nn.Module, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.backbone = model

    def get_embeddings(self, X):
        return self.backbone(X)

    def forward(self, X):
        return self.get_embeddings(X)


class ClassificationModel(EmbeddingModel):
    def __init__(self, model: nn.Module, classificator: nn.Module, embedding_size):
        super().__init__(model)
        self.classificator = classificator

    def replace_classificator(self, new_classificator: nn.Module, embedding_size):
        self.classificator = new_classificator

    def forward(self, X):
        embeddings = self.get_embeddings(X)
        return self.classificator(embeddings)


class MetricLearningModel(EmbeddingModel):
    pass
