"""Write a Supervisely project out as a YOLO OBB dataset.

``Project.to_yolo`` only knows detect/segment/pose (see
``supervisely/convert/image/yolo/yolo_helper.py``), so oriented detection needs
its own writer. The directory layout and ``data_config.yaml`` produced here are
deliberately identical to the SDK's, so everything downstream - the Ultralytics
``SettingsManager`` paths set in ``main.py``, TensorBoard, the experiment page -
sees the same structure it does for the other task types.

Only the label lines differ. YOLO OBB wants the four corners of each object,
``class x1 y1 x2 y2 x3 y3 x4 y4`` normalized to the image size, matching
``ultralytics.data.converter.convert_dota_to_yolo_obb``.
"""

import os
import shutil
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import yaml

import supervisely as sly
from supervisely._utils import generate_free_name
from supervisely.io.fs import get_file_name

from supervisely_integration import obb

DATA_CONFIG_NAME = "data_config.yaml"


def project_to_yolo_obb(
    project: sly.Project,
    dest_dir: str,
    val_datasets: Optional[List[str]] = None,
    progress_cb: Optional[Callable] = None,
) -> str:
    """Convert a Supervisely project to the YOLO OBB dataset layout.

    :param project: Project to convert, already split and filtered by the TrainApp.
    :param dest_dir: Destination directory; must not exist or must be empty.
    :param val_datasets: Names of the datasets to treat as validation. When
        omitted, an image is validation if it carries the ``val`` tag, which is
        how the SDK's own YOLO writer decides.
    :param progress_cb: Called with ``1`` per converted image.
    :returns: ``dest_dir``.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if len(os.listdir(dest_dir)) > 0:
        raise FileExistsError(f"Directory {dest_dir} is not empty.")

    class_names = [obj_class.name for obj_class in project.meta.obj_classes]
    _save_data_config(project.meta, dest_dir)

    split_dirs = {}
    for split in ("train", "val"):
        images_dir = dest_dir / "images" / split
        labels_dir = dest_dir / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        split_dirs[split] = (images_dir, labels_dir)

    used_names = set()
    total_boxes, total_dropped = 0, 0
    for dataset in project.datasets:
        dataset: sly.Dataset
        dataset_is_val = None if val_datasets is None else dataset.name in val_datasets

        for name in dataset.get_items_names():
            ann = sly.Annotation.load_json_file(dataset.get_ann_path(name), project.meta)
            is_val = dataset_is_val
            if is_val is None:
                is_val = ann.img_tags.get("val") is not None
            images_dir, labels_dir = split_dirs["val" if is_val else "train"]

            img_path = Path(dataset.get_img_path(name))
            img_name = generate_free_name(
                used_names,
                f"{dataset.short_name}_{img_path.name}",
                with_ext=True,
                extend_used_names=True,
            )
            shutil.copy2(img_path, images_dir / img_name)

            lines, dropped = _ann_to_obb_lines(ann, class_names)
            total_boxes += len(lines)
            total_dropped += dropped
            (labels_dir / f"{get_file_name(img_name)}.txt").write_text("\n".join(lines))

            if progress_cb is not None:
                progress_cb(1)

        sly.logger.info(f"Dataset '{dataset.short_name}' has been converted to YOLO OBB format.")

    if total_dropped > 0:
        sly.logger.warning(
            f"{total_dropped} label(s) fell outside their image and were skipped."
        )
    sly.logger.info(
        f"Project '{project.name}' has been converted to YOLO OBB format: "
        f"{total_boxes} oriented box(es) written to '{dest_dir}'."
    )
    return str(dest_dir)


def _ann_to_obb_lines(ann: sly.Annotation, class_names: List[str]) -> Tuple[List[str], int]:
    """Convert one annotation to YOLO OBB label lines, counting the boxes dropped."""
    height, width = ann.img_size
    image_size = np.array([width, height], dtype=np.float64)

    lines, dropped = [], 0
    for label in ann.labels:
        if label.obj_class.name not in class_names:
            continue
        class_idx = class_names.index(label.obj_class.name)

        for corners in obb.label_to_corners(label):
            corners = obb.clip_corners(corners, height, width)
            if corners is None:
                dropped += 1
                continue
            coords = " ".join(f"{value:.6g}" for value in (corners / image_size).reshape(-1))
            lines.append(f"{class_idx} {coords}")
    return lines, dropped


def _save_data_config(meta: sly.ProjectMeta, dest_dir: Path) -> None:
    """Write the same ``data_config.yaml`` the SDK writes for the other task types."""
    data_yaml = {
        "train": f"../{dest_dir.name}/images/train",
        "val": f"../{dest_dir.name}/images/val",
        "train_labels": f"../{dest_dir.name}/labels/train",
        "val_labels": f"../{dest_dir.name}/labels/val",
        "nc": len(meta.obj_classes),
        "names": [obj_class.name for obj_class in meta.obj_classes],
        "colors": [obj_class.color for obj_class in meta.obj_classes],
    }
    save_path = dest_dir / DATA_CONFIG_NAME
    with open(save_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=None)
    sly.logger.info(f"Data config file has been saved to {str(save_path)}")
