# flake8: noqa

import json
import os

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


def load_coco_json(path):
    """Read json with annotations.

    Args:
        path (str): path to .json file

    Raises:
        RuntimeError if .json file has no images
        RuntimeError if .json file has no categories

    Returns:
        images mapping and categories mapping
    """

    with open(path, "r") as in_file:
        content = json.load(in_file)

    if not len(content["images"]):
        raise RuntimeError(f"There is no image records in '{path}' file!")

    if not len(content["categories"]):
        raise RuntimeError(f"There is no categories in '{path}' file!")

    # image_id -> {
    #   file_name,
    #   height,
    #   width,
    #   annotations([{id, iscrowd, category_id, bbox}, ...])
    # }
    images = {}
    for record in content["images"]:
        images[record["id"]] = {
            "file_name": record["file_name"],
            "height": record["height"],
            "width": record["width"],
            "annotations": [],
        }

    categories = {}  # category_id -> name
    for record in content["categories"]:
        categories[record["id"]] = record["name"]

    for record in content["annotations"]:
        images[record["image_id"]]["annotations"].append(
            {
                "id": record["id"],
                "iscrowd": record["iscrowd"],
                "category_id": record["category_id"],
                "bbox": record["bbox"],
            }
        )

    return images, categories


def read_image(path):
    """Read image from given path.

    Args:
        path (str or Path): path to an image.

    Raises:
        FileNotFoundError when missing image file

    Returns:
        np.ndarray with image.
    """
    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(f"There is no '{path}'!")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def pixels_to_absolute(box, width, height):
    """Convert pixel coordinates to absolute scales ([0,1]).

    Args:
        box (Tuple[number, number, number, number]): bounding box coordinates,
            expected list/tuple with 4 int values (x, y, w, h).
        width (int): image width
        height (int): image height

    Returns:
        List[float, float, float, float] with absolute coordinates (x1, y1, x2, y2).
    """
    x, y, w, h = box
    return [x / width, y / height, (x + w) / width, (y + h) / height]


def clip(values, min_value=0.0, max_value=1.0):
    return [min(max(num, min_value), max_value) for num in values]


def change_box_order(boxes, order):
    """Change box order between
    (xmin, ymin, xmax, ymax) <-> (xcenter, ycenter, width, height).

    Args:
        boxes: (torch.Tensor or np.ndarray) bounding boxes, sized [N,4].
        order: (str) either "xyxy2xywh" or "xywh2xyxy".

    Returns:
        (torch.Tensor) converted bounding boxes, sized [N,4].
    """
    if order not in {"xyxy2xywh", "xywh2xyxy"}:
        raise ValueError("`order` should be one of 'xyxy2xywh'/'xywh2xyxy'!")

    concat_fn = torch.cat if isinstance(boxes, torch.Tensor) else np.concatenate

    a = boxes[:, :2]
    b = boxes[:, 2:]
    if order == "xyxy2xywh":
        return concat_fn([(a + b) / 2, b - a], 1)
    return concat_fn([a - b / 2, a + b / 2], 1)


class YOLOXDataset(Dataset):
    def __init__(self, coco_json_path, images_dir=None, transforms=None, max_objects_on_image=120):
        self.file = coco_json_path
        self.images_dir = images_dir
        self.transforms = transforms
        self.max_objects_on_image = max_objects_on_image

        self.images, self.categories = load_coco_json(coco_json_path)
        self.images_list = sorted(self.images.keys())

        self.class_to_cid = {
            cls_idx: cat_id for cls_idx, cat_id in enumerate(sorted(self.categories.keys()))
        }
        self.cid_to_class = {v: k for k, v in self.class_to_cid.items()}
        self.num_classes = len(self.class_to_cid)
        self.class_labels = [
            self.categories[self.class_to_cid[cls_idx]]
            for cls_idx in range(len(self.class_to_cid))
        ]

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        img_id = self.images_list[index]
        img_record = self.images[img_id]

        path = img_record["file_name"]
        if self.images_dir is not None:
            path = os.path.join(self.images_dir, path)
        image = read_image(path)

        boxes = []  # each element is a tuple of (x1, y1, x2, y2, "class")
        raw_annotations = img_record["annotations"][: self.max_objects_on_image]
        for annotation in raw_annotations:
            xyxy = pixels_to_absolute(
                annotation["bbox"], img_record["width"], img_record["height"]
            )
            xyxy = clip(xyxy, 0.0, 1.0)
            bbox_class = str(self.cid_to_class[annotation["category_id"]])
            boxes.append(xyxy + [str(bbox_class)])

        if self.transforms is not None:
            transformed = self.transforms(image=image, bboxes=boxes)
            image, boxes = transformed["image"], transformed["bboxes"]
        else:
            image = torch.from_numpy((image / 255.0).astype(np.float32)).permute(2, 0, 1)

        bboxes = np.zeros((self.max_objects_on_image, 4), dtype=np.float32)
        classes = np.zeros(self.max_objects_on_image, dtype=np.int32)
        for idx, (x1, y1, x2, y2, box_cls) in enumerate(boxes[: self.max_objects_on_image]):
            bboxes[idx, :] = [x1, y1, x2, y2]
            classes[idx] = int(box_cls)

        # scaling [0,1] -> [h, w]
        bboxes = bboxes * (image.size(1), image.size(2), image.size(1), image.size(2))
        bboxes = torch.from_numpy(bboxes)
        bboxes = change_box_order(bboxes, "xyxy2xywh")
        classes = torch.LongTensor(classes)

        return {
            "image": image,
            "boxes": bboxes,
            "classes": classes,
        }

    @staticmethod
    def collate_fn(batch):
        """
        Collect batch for YOLO X model.

        Args:
            batch (List[Dict[str, torch.Tensor]]):
                List with records from YOLOXDataset.

        Returns:
            images batch with shape [B, C, H, W]
            boxes with shape [B, MAX_OBJECTS, 4]
            classes with shape [B, MAX_OBJECTS,]
        """
        images, boxes, classes = [], [], []
        for item in batch:
            images.append(item["image"])
            boxes.append(item["boxes"])
            classes.append(item["classes"])

        images = torch.stack(images)
        boxes = torch.stack(boxes)
        classes = torch.stack(classes)

        return {"image": images, "bboxes": boxes, "labels": classes}