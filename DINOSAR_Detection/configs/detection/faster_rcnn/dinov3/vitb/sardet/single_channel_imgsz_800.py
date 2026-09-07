import math

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler

from ...base_dinov3 import (
    base_batch_size_for_lr,
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


image_size = 800
batch_size = 16

model.pixel_mean = [0.219 * 255.0]
model.pixel_std = [0.220 * 255.0]
model.input_format = "L"
model.backbone.net.img_size = image_size
model.backbone.net.in_chans = 1
model.backbone.net.n_storage_tokens = 4
model.backbone.square_pad = image_size

dataloader.image_size = image_size
dataloader.train.total_batch_size = batch_size
dataloader.train.mapper.image_format = "L"
dataloader.train.mapper.augmentations[1].short_edge_length = image_size
dataloader.train.mapper.augmentations[1].max_size = image_size
dataloader.test.mapper.augmentations[0].short_edge_length = image_size
dataloader.test.mapper.augmentations[0].max_size = image_size

train.init_checkpoint = (
    "weights/dinosar_b16_unisar7m_60e_detectron2.pth?matching_heuristics=False"
)
train.output_dir = checkpoint_output_dir(__file__, train.init_checkpoint)
optimizer.lr = 1e-4 * batch_size / base_batch_size_for_lr

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
