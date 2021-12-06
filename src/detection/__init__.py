import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


from catalyst.registry import Registry

from .callbacks import DetectionMeanAveragePrecision
from .custom_runner import YOLOXDetectionRunner
from catalyst.loggers import MLflowLogger
from .dataset import YOLOXDataset  
from .models import ( 
    yolo_x_tiny,
    yolo_x_small,
    yolo_x_medium,
    yolo_x_large,
    yolo_x_big,
)

# runers
Registry(YOLOXDetectionRunner)

# models
Registry(yolo_x_tiny)
Registry(yolo_x_small)
Registry(yolo_x_medium)
Registry(yolo_x_large)
Registry(yolo_x_big)

# callbacks
Registry(DetectionMeanAveragePrecision)

# datasets
Registry(YOLOXDataset)