import json
import os
import sys
from tqdm import tqdm
import xmltodict
import pandas as pd 

def load_annotations(annot_file, img_file):
    """Read annotations from file.

    Args:
        file (str): path to file.

    Returns:
        List[Dict[str, Any]] with bounding boxes
    """
    # with open(file, "r") as in_file:
    #     content = xmltodict.parse(in_file.read())
    # filename = annot_file[annot_file['file_name'] == ] # content["annotation"]["filename"]
    width = 6000
    height = 6000
    # import pdb;pdb.set_trace()
    # objects = content["annotation"]["object"]
    # objects = [objects] if isinstance(objects, dict) else objects
    annots = []
    for item in annot_file[annot_file['file_name'] == img_file].iterrows():
        x1 = item[1]['x_from']
        y1 = item[1]['y_from']
        x2 = item[1]['width']
        y2 = item[1]['height']
        annots.append(
            {
                "category": "walrus",
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x1 + x2),
                "y2": int(y1 + y2),
            }
        )
    return img_file, (width, height), annots


def main():
    """Convert data to COCO format."""
    imgs_dir, annots_dir, output_file = sys.argv[1], sys.argv[2], sys.argv[3]
    category2id = {"walrus": 1}
    categories = [{"id": cat_id, "name": cat_name} for cat_name, cat_id in category2id.items()]
    images = []
    annotations = []
    img_id = 1
    annot_id = 1
    annot_file = pd.read_csv("all_annotation_walrus_62k.csv")
    for img_file in tqdm(os.listdir(imgs_dir)):
        if not img_file.endswith(".jpg"):
            continue
        # annot_file = os.path.join(annots_dir, img_file[:-4] + ".xml")
        # import pdb;pdb.set_trace()
        filename, (width, height), annots = load_annotations(annot_file, img_file)
        # import pdb;pdb.set_trace()
        images.append({"id": img_id, "file_name": filename, "width": width, "height": height})
        for item in annots:
            cat_id = category2id[item["category"]]
            # import pdb;pdb.set_trace()
            x1, y1 = item["x1"], item["y1"]
            x2, y2 = item["x2"], item["y2"]
            area = (x2 - x1) * (y2 - y1)
            annotations.append(
                {
                    "id": annot_id,
                    "image_id": img_id,
                    "iscrowd": 0,
                    "area": area,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "category_id": cat_id,
                }
            )
            annot_id += 1
        img_id += 1

    with open(output_file, "w") as out_file:
        json.dump(
            {"categories": categories, "images": images, "annotations": annotations},
            out_file,
            indent=2,
        )


if __name__ == "__main__":
    main()