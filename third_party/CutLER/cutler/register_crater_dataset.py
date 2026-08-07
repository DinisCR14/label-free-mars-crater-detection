import os
from pathlib import Path

from detectron2.data.datasets import register_coco_instances

dataset_root = Path(os.environ.get("CRATER_DATASET_ROOT", "datasets/crater_dataset"))
annotation_root = Path(os.environ.get("CRATER_ANNOTATIONS_DIR", dataset_root / "annotations"))
initial_annotations = os.environ.get("CRATER_INITIAL_ANNOTATIONS", "train.json")


def annotation_path(filename: str) -> str:
    path = Path(filename)
    return str(path if path.is_absolute() else annotation_root / path)


def image_path(split: str) -> str:
    return str(dataset_root / split)

register_coco_instances(
    "crater_dataset_train",
    {},
    annotation_path(initial_annotations),
    #os.path.join(dataset_root, "annotations/train_i2edged_mefiltered_2000.json"),
    #os.path.join(dataset_root, "annotations/train_i2edged_mefiltered_2000_removed.json"),
    #os.path.join(dataset_root, "annotations/all_train_truegt_withseg.json"),
    #os.path.join(dataset_root, "annotations/train_i2edged_mefiltered_2000_all.json"),
    image_path("train"),
)

register_coco_instances(
    "crater_dataset_train_r1",
    {},
    annotation_path("train_r1.json"),
    image_path("train"),
)

register_coco_instances(
    "crater_dataset_train_r2",
    {},
    annotation_path("train_r2.json"),
    image_path("train"),
)

register_coco_instances(
    "crater_dataset_train_r3",
    {},
    annotation_path("train_r3.json"),
    image_path("train"),
)

register_coco_instances(
    "crater_dataset_train_r4",
    {},
    annotation_path("train_run18_r4_03.json"),
    image_path("train"),
)

register_coco_instances(
    "crater_dataset_train_r5",
    {},
    annotation_path("train_run11_r5_2000.json"),
    image_path("train"),
)

register_coco_instances(
    "crater_dataset_train_r6",
    {},
    annotation_path("train_run11_r6_2000.json"),
    image_path("train"),
)

register_coco_instances(
    "crater_dataset_train_r7",
    {},
    annotation_path("train_run12_r7.json"),
    image_path("train"),
)

register_coco_instances(
    "crater_dataset_val",
    {},
    annotation_path("val_truegt.json"),
    image_path("val"),
)

register_coco_instances(
    "crater_dataset_test",
    {},
    annotation_path("test_truegt_clipped.json"),
    image_path("test"),
)
