from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class AIRPolSARSegDataset(BaseSegDataset):
    METAINFO = dict(
        classes=("Industrial", "Natural", "Water", "Land_Use", "Housing", "Other"),
        palette=[
            [0, 0, 255],
            [0, 255, 0],
            [0, 255, 255],
            [255, 0, 0],
            [255, 255, 0],
            [255, 255, 255],
        ],
    )

    def __init__(self, img_suffix=".png", seg_map_suffix=".png", reduce_zero_label=False, **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs,
        )


@DATASETS.register_module()
class AIRPolSARSegWaterDataset(BaseSegDataset):
    METAINFO = dict(
        classes=("background", "Water"),
        palette=[[0, 0, 0], [0, 255, 255]],
    )

    def __init__(self, img_suffix=".png", seg_map_suffix=".png", reduce_zero_label=False, **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs,
        )
