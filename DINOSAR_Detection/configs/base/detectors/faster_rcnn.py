from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import (
    FastRCNNConvFCHead,
    FastRCNNOutputLayers,
    StandardROIHeads,
)

from configs.base.common import num_classes


model = model_zoo.get_config("common/models/mask_rcnn_fpn.py").model

model.roi_heads = L(StandardROIHeads)(
    num_classes=num_classes,
    batch_size_per_image=512,
    positive_fraction=0.25,
    proposal_matcher=L(Matcher)(
        thresholds=[0.5],
        labels=[0, 1],
        allow_low_quality_matches=False,
    ),
    box_in_features=["p2", "p3", "p4", "p5"],
    box_pooler=L(ROIPooler)(
        output_size=7,
        scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
        sampling_ratio=0,
        pooler_type="ROIAlignV2",
    ),
    box_head=L(FastRCNNConvFCHead)(
        input_shape=ShapeSpec(channels=256, height=7, width=7),
        conv_dims=[256, 256, 256, 256],
        fc_dims=[1024],
        conv_norm="LN",
    ),
    box_predictor=L(FastRCNNOutputLayers)(
        input_shape=ShapeSpec(channels=1024),
        test_score_thresh=0.05,
        box2box_transform=L(Box2BoxTransform)(weights=(10, 10, 5, 5)),
        num_classes="${..num_classes}",
    ),
)

model.proposal_generator.head.conv_dims = [-1, -1]

train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.checkpointer = dict(period=10**9, max_to_keep=1)
train.best_checkpointer = dict(
    enabled=True,
    metric="bbox/AP",
    mode="max",
    file_prefix="model_best",
)
train.keep_checkpoint_files = True
train.log_period = 100
train.seed = 42
