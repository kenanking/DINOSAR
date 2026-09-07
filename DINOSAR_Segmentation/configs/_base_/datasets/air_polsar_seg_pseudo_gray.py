custom_imports = dict(imports=["projects.dinosar_mmseg"], allow_failed_imports=False)

dataset_type = "AIRPolSARSegDataset"
data_root = "datasets/AIR-PolSAR-Seg-mmseg"
crop_size = (512, 512)

train_pipeline = [
    dict(type="LoadImageFromFile", color_type="grayscale"),
    dict(type="LoadAnnotations"),
    dict(type="RandomResize", scale=(512, 512), ratio_range=(0.75, 1.25), keep_ratio=True),
    dict(type="RandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5, direction="horizontal"),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(type="RandomRotate", prob=0.5, degree=(-15, 15)),
    dict(type="PackSegInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", color_type="grayscale"),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="InfiniteSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="train_set/images", seg_map_path="train_set/annotations"),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="test_set/images", seg_map_path="test_set/annotations"),
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(type="ClasswiseIoUMetric", iou_metrics=["mIoU"])
test_evaluator = val_evaluator

data_preprocessor = dict(
    type="SegDataPreProcessor",
    size=crop_size,
    mean=[0.219 * 255.0],
    std=[0.220 * 255.0],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255,
)
