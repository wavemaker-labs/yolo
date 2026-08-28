"""Shared pieces for YOLO oriented bounding box (OBB) support.

The Supervisely SDK ships an ``OrientedBBox`` geometry and a ``PredictionBBox``
that already carries an ``angle``, but it has no ``TaskType`` constant for
oriented detection. ``OBB_TASK_TYPE`` below is that missing name: it is what
``models.json`` stores in ``meta.task_type``, what the model selector groups
pretrained checkpoints by, and what the training app reports back to the
platform. Everything downstream treats it as an opaque string, so it is defined
once here and imported by both apps.

Angle convention
----------------
Ultralytics returns an oriented box as ``xywhr``: a ``w`` by ``h`` box centered
at ``(cx, cy)`` and rotated by ``r`` radians, whose corners are built as
``center + R(r) @ (+-w/2, +-h/2)`` (``ultralytics.utils.ops.xywhr2xyxyxyxy``).

``sly.OrientedBBox`` stores the *unrotated* box as ``top/left/bottom/right``
plus an angle, and rotates those corners around the box center with the very
same matrix (``OrientedBBox.calculate_rotated_corners``). Both work in a y-down
image frame, so ``r`` carries over untouched: no sign flip, no 90 degree offset.
"""

from typing import List, Optional

import cv2
import numpy as np

import supervisely as sly
from supervisely.convert.image.yolo.yolo_helper import (
    SLY_YOLO_TASK_TYPE_MAP as _SDK_TASK_TYPE_MAP,
)
from supervisely.nn.prediction_dto import PredictionBBox

OBB_TASK_TYPE = "oriented object detection"
YOLO_OBB_TASK = "obb"

# The SDK map only covers detect/segment/pose.
SLY_YOLO_TASK_TYPE_MAP = {**_SDK_TASK_TYPE_MAP, OBB_TASK_TYPE: YOLO_OBB_TASK}


def to_yolo_task(task_type: str) -> str:
    """Convert a Supervisely task type to the ``task`` string Ultralytics expects."""
    yolo_task = SLY_YOLO_TASK_TYPE_MAP.get(task_type)
    if yolo_task is None:
        raise ValueError(
            f"Unsupported task type: '{task_type}'. "
            f"Supported types: {', '.join(SLY_YOLO_TASK_TYPE_MAP)}"
        )
    return yolo_task


def prediction_from_xywhr(class_name: str, xywhr, score: float) -> PredictionBBox:
    """Convert one Ultralytics ``xywhr`` row into a prediction DTO.

    The DTO holds the box as it would be without rotation, plus the angle, which
    is how the SDK moves oriented boxes around (see ``PredictionBBox.angle`` and
    ``nn/inference/tracking/bbox_tracking.py``).
    """
    center_x, center_y, width, height, angle = (float(value) for value in xywhr)
    bbox_tlbr = [
        round(center_y - height / 2),
        round(center_x - width / 2),
        round(center_y + height / 2),
        round(center_x + width / 2),
    ]
    return PredictionBBox(class_name, bbox_tlbr, score, angle)


def rotated_corners(top, left, bottom, right, angle) -> np.ndarray:
    """Return the four corners of an oriented box as float ``(x, y)`` pairs.

    Mirrors ``OrientedBBox.calculate_rotated_corners``, but keeps sub-pixel
    precision (the SDK truncates through integer ``PointLocation``) and pivots on
    the true midpoint instead of the SDK's floored one.
    """
    corners = np.array(
        [[left, top], [right, top], [right, bottom], [left, bottom]], dtype=np.float64
    )
    center = np.array([(left + right) / 2.0, (top + bottom) / 2.0])
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return (corners - center) @ rotation.T + center


def label_to_corners(label: sly.Label) -> List[np.ndarray]:
    """Fit oriented boxes to a label and return their corners.

    Rectangles keep their shape at angle 0, oriented boxes are used as-is, and
    anything with an outline (polygons, and masks by way of polygons) gets the
    minimum-area rotated rectangle around it. A mask that falls apart into
    several disjoint pieces yields one box per piece.
    """
    geometry = label.geometry
    # OrientedBBox subclasses Rectangle, so it has to be tested first.
    if isinstance(geometry, sly.OrientedBBox):
        return [
            rotated_corners(
                geometry.top, geometry.left, geometry.bottom, geometry.right, geometry.angle
            )
        ]
    if isinstance(geometry, sly.Rectangle):
        return [rotated_corners(geometry.top, geometry.left, geometry.bottom, geometry.right, 0.0)]
    if isinstance(geometry, sly.Polygon):
        return [_min_area_corners(geometry.exterior)]

    polygon_class = label.obj_class.clone(geometry_type=sly.Polygon)
    try:
        converted_labels = label.convert(polygon_class)
    except Exception:
        sly.logger.warning(
            f"Label of class '{label.obj_class.name}' has geometry "
            f"'{geometry.geometry_name()}', which cannot be converted to an oriented box. "
            "Skipping it.",
            exc_info=True,
        )
        return []
    return [
        _min_area_corners(converted.geometry.exterior)
        for converted in converted_labels
        if isinstance(converted.geometry, sly.Polygon) and len(converted.geometry.exterior) >= 3
    ]


def clip_corners(corners: np.ndarray, height: int, width: int) -> Optional[np.ndarray]:
    """Clip an oriented box into the image, or return ``None`` if it misses it.

    Rotating a box can push its corners past the image edge, and Ultralytics
    rejects a label file whose coordinates stray more than 1% outside ``[0, 1]``,
    dropping every box in it. Clamping each coordinate is exactly what
    Ultralytics does to oriented boxes that augmentation cuts off
    (``Instances.clip``), and it re-fits the rotated box with ``cv2.minAreaRect``
    afterwards (``ops.xyxyxyxy2xywhr``), so a clamped quad is the input it
    already expects.
    """
    xs, ys = corners[:, 0], corners[:, 1]
    if xs.max() <= 0 or ys.max() <= 0 or xs.min() >= width or ys.min() >= height:
        return None
    if xs.min() >= 0 and ys.min() >= 0 and xs.max() <= width and ys.max() <= height:
        return corners
    return np.stack([xs.clip(0, width), ys.clip(0, height)], axis=1)


def _min_area_corners(points) -> np.ndarray:
    """Minimum-area rotated rectangle around a list of ``PointLocation``s."""
    return _fit_min_area_box(np.array([[p.col, p.row] for p in points]))


def _fit_min_area_box(points: np.ndarray) -> np.ndarray:
    return cv2.boxPoints(cv2.minAreaRect(points.astype(np.float32))).astype(np.float64)
