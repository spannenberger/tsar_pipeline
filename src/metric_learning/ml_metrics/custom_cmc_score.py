from sklearn.neighbors import KNeighborsClassifier
from catalyst.metrics._cmc_score import CMCMetric
from typing import List
import torch


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

