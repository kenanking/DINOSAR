import math

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler

from ...base_dinov3 import (
    batch_size,
    checkpoint_output_dir,
    epochs,
    milestone_epochs,
    model,
    optimizer,
    train,
    warmup_steps,
)
from configs.base.datasets.sardet import dataloader, train_images


model.pixel_mean = [0.219 * 255.0]
model.pixel_std = [0.220 * 255.0]
model.input_format = "L"
model.backbone.net.in_chans = 1
model.backbone.net.n_storage_tokens = 4

dataloader.train.mapper.image_format = "L"

train.init_checkpoint = (
    "weights/dinosar_b16_unisar7m_60e_detectron2.pth?matching_heuristics=False"
)
train.output_dir = checkpoint_output_dir(__file__, train.init_checkpoint)

iters_per_epoch = math.ceil(train_images / batch_size)
train.max_iter = iters_per_epoch * epochs
train.eval_period = iters_per_epoch

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[iters_per_epoch * e for e in milestone_epochs],
        num_updates=train.max_iter,
    ),
    warmup_length=warmup_steps / train.max_iter,
    warmup_factor=0.001,
)
