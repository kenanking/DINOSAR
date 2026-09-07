# AIR-PolSARSeg Preprocessing

The project uses one canonical image product for all AIR-PolSARSeg experiments:

```text
R = (HV + VH) / 2
G = HH
B = VV
```

Each channel is independently converted to `uint8` by:

1. clipping negative values to zero,
2. applying `log1p`,
3. scaling each image channel by the 1st and 99th percentiles,
4. clipping to `[0, 255]`,
5. inverting intensity as `255 - value`.

Generate the MMSeg layout with:

```bash
python scripts/prepare_airseg_dataset.py \
  --raw-root datasets/Raw_AIR-PolarSAR-Seg \
  --out-root datasets/AIR-PolSAR-Seg-mmseg \
  --overwrite
```

The generated directory is:

```text
datasets/AIR-PolSAR-Seg-mmseg/
  train_set/images/*.png
  train_set/annotations/*.png
  train_set_water/annotations/*.png
  test_set/images/*.png
  test_set/annotations/*.png
  test_set_water/annotations/*.png
  meta.json
```


Single-channel models, such as DINOSAR, use the same PNG files but read them with `LoadImageFromFile(color_type="grayscale")` and DINOSAR's pretraining normalization in uint8 scale: `mean=[0.219 * 255]`, `std=[0.220 * 255]`.

This conversion is a local, auditable approximation of the processed AIR-PolSARSeg images used by prior three-channel baselines; the original TIFF to PNG conversion script is not published by those baselines, so `meta.json` should be kept with every generated dataset copy.

Label conversion:

| RGB color         | Class id | Class      |
| ----------------- | -------: | ---------- |
| `[0, 0, 255]`     |        0 | Industrial |
| `[0, 255, 0]`     |        1 | Natural    |
| `[0, 255, 255]`   |        2 | Water      |
| `[255, 0, 0]`     |        3 | Land_Use   |
| `[255, 255, 0]`   |        4 | Housing    |
| `[255, 255, 255]` |        5 | Other      |
| `[0, 0, 0]`       |      255 | ignore     |

The water task maps cyan water to class `1`, all other known classes to background `0`, and black pixels to ignore `255`.
