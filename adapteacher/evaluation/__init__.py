# Copyright (c) Facebook, Inc. and its affiliates.
from .coco_evaluation import COCOEvaluator
from .pascal_voc_evaluation import PascalVOCDetectionEvaluator
from .sem_eval import newSemsSegEvaluator

# __all__ = [k for k in globals().keys() if not k.startswith("_")]

__all__ = [
    "COCOEvaluator",
    "PascalVOCDetectionEvaluator",
    "newSemsSegEvaluator",
]
