import os
from pathlib import Path

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from omegaconf import OmegaConf

from ..common import batch_size, image_size


LOCAL_DATASET_ROOT = Path("/path/to/SARDet100K_coco_annotation")
dataset_root = Path(os.environ.get("DATASET_ROOT", str(LOCAL_DATASET_ROOT)))

register_coco_instances(
    "sardet-100k_train",
    {},
    str(dataset_root / "Annotations/train.json"),
    str(dataset_root / "Images/train"),
)
register_coco_instances(
    "sardet-100k_test",
    {},
    str(dataset_root / "Annotations/test.json"),
    str(dataset_root / "Images/test"),
)

dataloader = OmegaConf.create()
dataloader.image_size = image_size
dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="sardet-100k_train"),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.RandomFlip)(horizontal=True),
            # Keep detection geometry valid for the non-square SARDet images and
            # leave fixed-square padding to Detectron2's ImageList/model preprocessing.
            L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
        ],
        image_format="RGB",
        use_instance_mask=False,
        use_keypoint=False,
    ),
    total_batch_size=batch_size,
    num_workers=8,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(
        names="sardet-100k_test",
        filter_empty=False,
    ),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            # Do not crop/pad in the dataset mapper; otherwise postprocess treats
            # padded pixels as real image area and rescales boxes incorrectly.
            L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
        ],
        image_format="${...train.mapper.image_format}",
        use_instance_mask=False,
        use_keypoint=False,
    ),
    num_workers=8,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)

train_images = len(get_detection_dataset_dicts("sardet-100k_train"))
