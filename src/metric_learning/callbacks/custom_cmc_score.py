from catalyst.dl import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.registry import Registry
import torch
from typing import List, Iterable
from catalyst.metrics._cmc_score import CMCMetric
from catalyst.callbacks.metric import LoaderMetricCallback
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class CustomCMCScoreCallback(LoaderMetricCallback):
    def __init__(
        self,
        embeddings_key: str,
        labels_key: str,
        is_query_key: str,
        topk_args: Iterable[int] = None,
        prefix: str = None,
        suffix: str = None,
    ):
        """Init."""
        super().__init__(
            metric=CustomCMCMetric(
                embeddings_key=embeddings_key,
                labels_key=labels_key,
                is_query_key=is_query_key,
                topk_args=topk_args,
                prefix=prefix,
                suffix=suffix,
            ),
            input_key=[embeddings_key, is_query_key],
            target_key=[labels_key],
        )


class CustomCMCMetric(CMCMetric):
    def compute(self) -> List[float]:
        """
        Compute cmc@k metrics with all the accumulated data for all k.
        Returns:
            list of metrics values
        """
        
        query_mask = (self.storage[self.is_query_key] == 1).to(torch.bool)

        embeddings = self.storage[self.embeddings_key].float()
        labels = self.storage[self.labels_key]

        query_embeddings = embeddings[query_mask]
        query_labels = labels[query_mask]

        gallery_embeddings = embeddings[~query_mask]
        gallery_labels = labels[~query_mask]

        metrics = []
        value = custom_cmc_score(
            query_embeddings=query_embeddings,
            gallery_embeddings=gallery_embeddings,
            query_labels=query_labels,
            gallery_labels=gallery_labels,
        )
        metrics.append(value)

        return metrics

def custom_cmc_score_count(
    gallery_embeddings: torch.Tensor, 
    gallery_labels: torch.Tensor, 
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor) -> float:

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(gallery_embeddings, gallery_labels)
    preds = torch.Tensor(knn.predict(query_embeddings))
    positive = preds == query_labels
    return (positive.sum()/len(positive)).item()


def custom_cmc_score(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor) -> float:

    return custom_cmc_score_count(gallery_embeddings, gallery_labels, query_embeddings, query_labels)

