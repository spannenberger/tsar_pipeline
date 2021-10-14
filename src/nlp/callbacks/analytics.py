from catalyst.dl import Callback, CallbackOrder
from catalyst.core.runner import IRunner
from catalyst.registry import Registry
import torch
import pandas as pd
from transformers import BertTokenizer
from scipy.spatial.distance import cosine
from tqdm import tqdm


@Registry
class DistanceAnalyse(Callback):
    def __init__(self, data_path, similary_file_path):
        self.data_path = data_path
        self.similary_file_path = similary_file_path
        super().__init__(CallbackOrder.ExternalExtra)

    def on_experiment_end(self, state: IRunner):
        if state.is_valid_loader:
            df = pd.read_csv(self.data_path)
            sents = list(set(df["user_story"]))
            tokenizer = BertTokenizer.from_pretrained('sberbank-ai/ruBert-large')
            distance_batch = tokenizer(sents, padding=True, truncation=True, max_length=100, return_tensors='pt')
            model = state.model
            print("Eval Model")
            with torch.no_grad():
                outputs = model(**distance_batch)
            # import pdb; pdb.set_trace()
            sentence_embedding = outputs[0]

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

            # Считаем и выводим расстояния между user story
            print("\nСчитаем и записываем расстояния между user story")
            with open(self.similary_file_path, "w", encoding="utf-8") as dist:
                for i in tqdm(similarity):
                    dist.write(f"Более похожие на предложение\n\n {i}\nПредложения:\n\n")
                    for idx, j in enumerate(similarity[i]):
                        if idx == 5:
                            break
                        dist.write(f"{j[0]} Расстояние: {j[1].item()}\n\n")
                    dist.write("\n\n\n")
