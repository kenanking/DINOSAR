# Visualization examples

`ship.png`, `aircraft.png`, `bridge.png`, and `car.png` are the existing DINOSAR
visualization examples. Three additional HOSS ship crops are copied unchanged
from the original experiment archive:

| File | Original archive member |
| --- | --- |
| `hoss_0401_s03c1_SAR.tif` | `HOSS/bounding_box_train/0401_s03c1_SAR.tif` |
| `hoss_0610_s07c2_SAR.tif` | `HOSS/bounding_box_train/0610_s07c2_SAR.tif` |
| `hoss_0647_s07c2_SAR.tif` | `HOSS/bounding_box_train/0647_s07c2_SAR.tif` |

The HOSS crops were selected in the original similarity/PCA experiment.
They are illustrative inputs, not an evaluation split. The notebook applies
optional percentile contrast scaling at runtime and preserves aspect ratio.
Original dataset attribution and terms apply to the HOSS samples.
