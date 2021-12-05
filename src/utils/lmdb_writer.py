import numpy as np
import pyxis as px
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import cv2
import albumentations as A
from tqdm import tqdm
import yaml 
from pathlib import Path

class LMDBDataset(Dataset):
    """ LMDB data class """
    def __init__(self, root, loader) -> None:
        self._dataset = ImageFolder(root=root, loader=loader)
        super().__init__()
    
    def __getitem__(self, idx):
        item = {}
        item["sample"] = self._dataset[idx][0]
        item["target"] = self._dataset[idx][1]
        item["image_name"] = self._dataset.imgs[idx][0]
        return item

    def __len__(self):
        return len(self._dataset)
        
if __name__ == "__main__":
    # read config
    with open("./src/utils/lmdb_config.yml", "r") as config:
        config_dict = yaml.load(config, Loader=yaml.FullLoader)

    width = config_dict["width"]
    height = config_dict["height"]
    channels = config_dict["channels"]
    dataset_path = config_dict["dataset_path"]
    num_workers = config_dict["num_workers"]
    batch_size = config_dict["batch_size"]
    pin_memory = config_dict["pin_memory"]
    output_dataset_path = Path(config_dict["output_dataset_path"])
    output_dataset_path.mkdir(parents=True, exist_ok=True)

    transforms = A.Compose([
        A.Resize(height, width),
        ])

    folders_list = ["/val", "/base", "/train"]

    for folder in folders_list:
        dataset = LMDBDataset(dataset_path + folder, loader=lambda x: transforms(image=cv2.imread(x))["image"])
        data_count = len(dataset)

        loader = DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, pin_memory=pin_memory)

        db = px.Writer(
            dirpath = str(output_dataset_path) + folder,
            map_size_limit=int(width*height*channels*data_count*1.5/2**20)
            )

        for i in tqdm(loader):
            data = {
                "image": i["sample"].numpy(),
                "target": i["target"].view(-1, 1).numpy(),
                "image_name": np.array(i["image_name"])
                }
            db.put_samples(data)
        db.close()
