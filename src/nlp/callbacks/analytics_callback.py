from catalyst.dl import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.registry import Registry
from sklearn.cluster import OPTICS
from scipy.spatial.distance import cosine
import pandas as pd
from pathlib import Path
import torch
from tqdm import tqdm
from dataset import NLPDataset
from torch.utils.data import DataLoader

@Registry
class AnalyticsDistanceCallback(Callback):

    def __init__(self, analytics_data: str, similarity_file_path: str, cluster_file_path: str):
        super().__init__(CallbackOrder.ExternalExtra)
        self.similarity_file_path = Path(similarity_file_path)
        self.cluster_file_path = Path(cluster_file_path)
        self.analytics_data = Path(analytics_data)
        self.similarity_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.cluster_file_path.parent.mkdir(parents=True, exist_ok=True)

    def on_experiment_end(self, state: IRunner):
        """ @TODO Docs
        """
        model = state.model.backbone
        tokenizer = state.tokenizer
        df = pd.read_csv(str(self.analytics_data))
        sents = list(set(df["user_story"]))
        distance_batch = tokenizer(sents, padding=True, truncation=True, max_length=100, return_tensors='pt')
        dataset = NLPDataset(**distance_batch ,labels=False)
        loader = DataLoader(dataset, shuffle=False, batch_size=1)
        outputs_final = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(loader):
                batch = state.engine.sync_device(tensor_or_module=batch)
                outputs = model(**batch)
                outputs_final.append(outputs.last_hidden_state.mean(axis=1).detach().cpu())
        sentence_embedding = torch.squeeze(torch.stack(outputs_final))
        sentence_embedding = sentence_embedding.view(-1, sentence_embedding.shape[1])
        heat=[]
        for i in sentence_embedding:
            line=[]
            for j in sentence_embedding:
                line.append(cosine(i,j))
            heat.append(torch.Tensor(line))
        heat = torch.stack(heat)

        similarity = {}
        for idx, i in enumerate(heat):
            similarity[sents[idx]]=[]
            for idx1, j in enumerate(i):
                if idx!=idx1:
                    similarity[sents[idx]].append([sents[idx1],j])

        for i in similarity:
            similarity[i] = sorted(similarity[i], key=lambda x: x[1])

        print("Write distances into txt file:\n")
        # Считаем и выводим расстояния между user story
        with open(self.similarity_file_path, "w", encoding="utf-8") as dist:
            for i in tqdm(similarity):
                dist.write(f"Более похожие на предложение\n\n {i}\nПредложения:\n\n")
                for idx, j in enumerate(similarity[i]):
                    if idx == 5:
                        break
                    dist.write(f"{j[0]} Расстояние: {j[1].item()}\n\n")
                dist.write("\n\n\n")
        print("Clustering user story:\n")
        # Кластеризируем user story
        clustering = OPTICS(eps=0.5, min_samples=2, metric='cosine').fit(sentence_embedding)
        print(f"Список всех кластеров: "+ ", ".join([str(x) for x in list(set(clustering.labels_))]))

        clusters = {}
        for idx, item in enumerate(clustering.labels_):
            if item in clusters:
                clusters[item].append(sents[idx])
            else:
                clusters[item] = [sents[idx]]

        # Добавить запись кластеров в csv для логгирования
        pd.DataFrame([pd.Series(clusters[x]) for x in clusters]).T.fillna('-').to_csv(self.cluster_file_path, index=False)
