from mean_average_precision import MetricBuilder
import numpy as np

import torch
import torch.nn.functional as F

from catalyst.core.callback import Callback, CallbackOrder
from .utils import change_box_order, nms_filter

def process_yolo_x_output(
    predicted_tensor,
    gt_boxes,
    gt_labels,
    iou_threshold=0.5,
):
    """Generate bbox and classes from YOLO-X model outputs.

    Args:
        predicted_tensor (torch.Tensor): model outputs,
            expected shapes [batch, num anchors, 4 + 1 + num classes].
        gt_boxes (torch.Tensor): ground truth bounding boxes,
            expected shapes [batch, num anchors, 4].
        gt_labels (torch.Tensor): ground truth bounding box labels,
            expected shape [batch, num anchors].
        iou_threshold (float): IoU threshold to use in NMS.
            Default is ``0.5``.

    Yields:
        predicted sample (np.ndarray) and ground truth sample (np.ndarray)
    """
    batch_size = predicted_tensor.size(0)
    outputs = predicted_tensor.detach().cpu().numpy()

    _pred_boxes = outputs[:, :, :4]
    _pred_confidence = outputs[:, :, 4]
    _pred_cls = np.argmax(outputs[:, :, 5:], -1)

    _gt_boxes = gt_boxes.cpu().numpy()
    _gt_classes = gt_labels.cpu().numpy()
    _gt_boxes_mask = _gt_boxes.sum(axis=2) > 0

    for i in range(batch_size):
        # build predictions
        sample_bboxes = change_box_order(_pred_boxes[i], "xywh2xyxy")
        sample_bboxes, sample_classes, sample_confs = nms_filter(
            sample_bboxes, _pred_cls[i], _pred_confidence[i], iou_threshold=iou_threshold
        )
        pred_sample = np.concatenate(
            [sample_bboxes, sample_classes[:, None], sample_confs[:, None]], -1
        )
        pred_sample = pred_sample.astype(np.float32)

        # build ground truth
        sample_gt_mask = _gt_boxes_mask[i]
        sample_gt_bboxes = change_box_order(_gt_boxes[i][sample_gt_mask], "xywh2xyxy")
        sample_gt_classes = _gt_classes[i][sample_gt_mask]
        gt_sample = np.zeros((sample_gt_classes.shape[0], 7), dtype=np.float32)
        gt_sample[:, :4] = sample_gt_bboxes
        gt_sample[:, 4] = sample_gt_classes

        yield pred_sample, gt_sample


class DetectionMeanAveragePrecision(Callback):
    """Compute mAP for Object Detection task."""

    def __init__(
        self,
        num_classes=1,
        metric_key="mAP",
        output_type="ssd",
        iou_threshold=0.5,
        confidence_threshold=0.5,
    ):
        """
        Args:
            num_classes (int): Number of classes.
                Default is ``1``.
            metric_key (str): name of a metric.
                Default is ``"mAP"``.
            output_type (str): model output type. Valid values are ``"ssd"`` or
                ``"centernet"`` or ``"yolo-x"``.
                Default is ``"ssd"``.
            iou_threshold (float): IoU threshold to use in NMS.
                Default is ``0.5``.
            confidence_threshold (float): confidence threshold,
                proposals with lover values than threshold will be ignored.
                Default is ``0.5``.
        """
        super().__init__(order=CallbackOrder.Metric)
        assert output_type in ("ssd", "centernet", "yolo-x")

        self.num_classes = num_classes
        self.metric_key = metric_key
        self.output_type = output_type
        self.iou_threshold = iou_threshold
        self.confidence_threshold = confidence_threshold

        self.metric_fn = MetricBuilder.build_evaluation_metric(
            "map_2d", async_mode=False, num_classes=num_classes
        )

    def on_loader_start(self, runner: "IRunner"):  # noqa: D102, F821
        if not runner.is_valid_loader:
            return
        self.metric_fn.reset()

    def on_batch_end(self, runner: "IRunner"):  # noqa: D102, F821
        if not runner.is_valid_loader:
            return
        if self.output_type == "yolo-x":
            p_tensor = runner.batch["predicted_tensor"]
            gt_box = runner.batch["bboxes"]
            gt_labels = runner.batch["labels"]
            for predicted_sample, ground_truth_sample in process_yolo_x_output(
                p_tensor, gt_box, gt_labels, iou_threshold=self.iou_threshold
            ):
                self.metric_fn.add(predicted_sample, ground_truth_sample)

    def on_loader_end(self, runner: "IRunner"):  # noqa: D102, F821
        if not runner.is_valid_loader:
            return
        map_value = self.metric_fn.value()["mAP"]
        runner.loader_metrics[self.metric_key] = map_value