_base_ = [
    "../_base_/datasets/air_polsar_seg_pseudo_gray.py",
    "../_base_/runtime.py",
    "../_base_/schedules/airseg_72ep.py",
]

custom_imports = dict(imports=["projects.dinosar_mmseg"], allow_failed_imports=False)

crop_size = (512, 512)
norm_cfg = dict(type="SyncBN", requires_grad=True)
dinosar_mean = 0.219 * 255.0
dinosar_std = 0.220 * 255.0
data_preprocessor = dict(
    type="SegDataPreProcessor",
    size=crop_size,
    mean=[dinosar_mean],
    std=[dinosar_std],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
)

model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="DINOv3VisionTransformer",
        img_size=512,
        patch_size=16,
        in_chans=1,
        input_adapter="none",
        drop_path_rate=0.0,
        out_indices=(3, 5, 7, 11),
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        use_checkpoint=False,
        n_storage_tokens=4,
        fp32_attention=True,
        pretrained="experiments/weights/DINOSAR/dinosar_b16_unisar7m_60e.pth",
    ),
    decode_head=dict(
        type="UPerHead",
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        channels=512,
        num_classes=6,
        ignore_index=255,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            dict(type="DiceLoss", use_sigmoid=False, activate=True, reduction="mean", naive_dice=False, loss_weight=0.5),
        ],
    ),
    auxiliary_head=dict(
        type="FCNHead",
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=0.4),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="slide", stride=(384, 384), crop_size=crop_size),
)

optim_wrapper = dict(
    type="AmpOptimWrapper",
    optimizer=dict(type="AdamW", lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05),
    constructor="LayerDecayOptimizerConstructor_DINOv3",
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9),
    loss_scale="dynamic",
)

work_dir = "experiments/work_dirs/dinosar_vitb_upernet_airseg_pseudo_gray_60e_72ep"
