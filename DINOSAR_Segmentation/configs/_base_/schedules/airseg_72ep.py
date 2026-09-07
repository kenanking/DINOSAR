iters_per_epoch = 88
max_iters = 72 * iters_per_epoch
warmup_iters = iters_per_epoch

train_cfg = dict(type="IterBasedTrainLoop", max_iters=max_iters, val_interval=iters_per_epoch)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-6, by_epoch=False, begin=0, end=warmup_iters),
    dict(
        type="CosineAnnealingLR",
        eta_min=0.0,
        T_max=max_iters - warmup_iters,
        begin=warmup_iters,
        end=max_iters,
        by_epoch=False,
    ),
]
