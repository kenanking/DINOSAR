from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from fvcore.common.param_scheduler import MultiStepParamScheduler

from .single_channel import (
    checkpoint_output_dir,
    dataloader,
    iters_per_epoch,
    model,
    optimizer,
    train,
    warmup_steps,
)


epochs = 36
milestone_epochs = [28, 33]

train.output_dir = checkpoint_output_dir(__file__, train.init_checkpoint)
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
