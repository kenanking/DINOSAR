from functools import partial

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.modeling import SimpleFeaturePyramid
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from configs.base.common import (
    base_batch_size_for_lr,
    batch_size,
    checkpoint_output_dir,
    epochs,
    image_size,
    milestone_epochs,
    warmup_steps,
)
from configs.base.detectors.faster_rcnn import model, train
from models.Dinov3_ViT import DINOv3ViTBackbone, get_vit_lr_decay_rate_for_dinov3


model.pixel_mean = [x * 255.0 for x in [0.185, 0.185, 0.185]]
model.pixel_std = [x * 255.0 for x in [0.220, 0.220, 0.220]]
model.input_format = "RGB"

model.backbone = L(SimpleFeaturePyramid)(
    net=L(DINOv3ViTBackbone)(
        arch="vit_base",
        out_feature="last_feat",
        img_size=image_size,
        in_chans=3,
        n_storage_tokens=4,
        pretrained=False,
    ),
    in_feature="${.net.out_feature}",
    out_channels=256,
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    top_block=L(LastLevelMaxPool)(),
    norm="LN",
    square_pad=image_size,
)

train.init_checkpoint = "weights/dinosar_b16_unisar7m_60e_detectron2.pth?matching_heuristics=False"
train.output_dir = checkpoint_output_dir(__file__, train.init_checkpoint)

optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.lr = 1e-4 * batch_size / base_batch_size_for_lr
optimizer.params.lr_factor_func = partial(
    get_vit_lr_decay_rate_for_dinov3,
    num_layers=12,
    lr_decay_rate=0.7,
)
optimizer.params.overrides = {"rope_embed": {"weight_decay": 0.0}}
optimizer.weight_decay = 0.05
