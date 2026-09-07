# Dataset Preparation

Prepare UniSAR-7M for pretraining and the classification benchmarks below
for evaluation. Other tasks document their data in their own READMEs.

## Pretraining Dataset

The final models use **UniSAR-7M**, a corpus of 7,047,666 SAR images.
Download all 18 archive parts (`UniSAR-7M.tar.gz.part_000` through
`UniSAR-7M.tar.gz.part_017`) from the
[Hugging Face dataset repository](https://huggingface.co/datasets/YTang/UniSAR-7M)
or [Google Drive](https://drive.google.com/drive/folders/16hax44FA9aAQfiypqBPfsSandtYl_auc).
The Dataset Card describes corpus composition and preprocessing.

Reconstruct the archive by concatenating the parts in index order, then extract it:

```bash
cat UniSAR-7M.tar.gz.part_* > UniSAR-7M.tar.gz
tar -xzf UniSAR-7M.tar.gz
```

Point the `data.data_path` entry of the pretraining configs at the directory containing the corpus images (the extracted top-level folder). The dataloader scans the dataset root recursively, so the internal directory layout does not matter; the image index (`index.txt`) used by the dataloader is built automatically on first use.

## Evaluation Datasets

The current evaluation datasets are aligned with the [SAR-JEPA](https://github.com/waterdisappear/SAR-JEPA) benchmark at the source-dataset level, but they are not exact protocol copies in every case. Specifically, `MSTAR` follows the standard `SOC10` protocol, `FUSAR_Ship` uses the same raw sources and class mapping but removes exact-content duplicates, and `SAR-ACD` retains the original six-class release rather than the five-class subset adopted by SAR-JEPA. All evaluation datasets currently documented below belong to classification tasks.

### Evaluation Preprocessing

All evaluation datasets use a unified preprocessing pipeline: the input image is converted to the model's expected channel count, then resized so its longest side matches the target input size (default 224) while preserving the aspect ratio. The shorter side is zero-padded to produce a square output. This avoids any content loss from center-cropping and ensures fair evaluation across datasets with variable image sizes. The transform is implemented by `ResizeAndPad` in `dinosar/evaluate.py` and applied identically to kNN, linear probe, and few-shot evaluations.

### Classification

#### MSTAR

To reconstruct `MSTAR`, download the official public release and place the prepared dataset under `experiments/datasets/evaluation/MSTAR`, organized as a flat `ImageFolder` layout:

```text
train/<class>/*
test/<class>/*
```

We follow the standard `SOC10` protocol, using the `17_DEG` subset for training and the `15_DEG` subset for testing. The benchmark contains 10 classes: `2S1`, `BMP2`, `BRDM2`, `BTR60`, `BTR70`, `D7`, `T62`, `T72`, `ZIL131`, and `ZSU234`. For `BMP2`, `BTR70`, and `T72`, only the standard single-serial subsets are retained, namely `SN_9563`, `SN_C71`, and `SN_132`. All images should be placed directly under the class directory (e.g. `train/BMP2/*.jpeg`), not nested in serial-number subdirectories. The seven classes already distributed as image files are taken directly from the public mixed-target packages, whereas `BMP2`, `BTR70`, and `T72` are reconstructed from the public target-chip package. Original filenames and suffixes are preserved whenever the source already provides displayable images; non-image target-chip files are converted into viewable image files during preprocessing.

**Statistics**

<table>
  <thead>
    <tr>
      <th>Split</th>
      <th>Class</th>
      <th>Count</th>
      <th>Class</th>
      <th>Count</th>
      <th>Class</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">train</td>
      <td>2S1</td>
      <td>299</td>
      <td>BMP2</td>
      <td>233</td>
      <td>BRDM2</td>
      <td>298</td>
    </tr>
    <tr>
      <td>BTR60</td>
      <td>256</td>
      <td>BTR70</td>
      <td>233</td>
      <td>D7</td>
      <td>299</td>
    </tr>
    <tr>
      <td>T62</td>
      <td>299</td>
      <td>T72</td>
      <td>232</td>
      <td>ZIL131</td>
      <td>299</td>
    </tr>
    <tr>
      <td>ZSU234</td>
      <td>299</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="4">test</td>
      <td>2S1</td>
      <td>274</td>
      <td>BMP2</td>
      <td>195</td>
      <td>BRDM2</td>
      <td>274</td>
    </tr>
    <tr>
      <td>BTR60</td>
      <td>195</td>
      <td>BTR70</td>
      <td>196</td>
      <td>D7</td>
      <td>274</td>
    </tr>
    <tr>
      <td>T62</td>
      <td>273</td>
      <td>T72</td>
      <td>196</td>
      <td>ZIL131</td>
      <td>274</td>
    </tr>
    <tr>
      <td>ZSU234</td>
      <td>274</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>

#### FUSAR_Ship

To rebuild `FUSAR_Ship`, download the full official FUSAR package from [https://radars.ac.cn/web/data/getData?dataType=FUSAR](https://radars.ac.cn/web/data/getData?dataType=FUSAR) and use the `.rar` release rather than the reduced ship-only archive. The full package contains both `ShipCategory` and `MarineCategory`, and both are required to reconstruct the evaluation benchmark. Place the prepared dataset under `experiments/datasets/evaluation/FUSAR_Ship`, organized as follows:

```text
train/<class>/*
test/<class>/*
```

The benchmark is defined as a 10-class classification task under a `train/test` split, with classes `Bridges`, `Cargo`, `CoastalLands_island`, `Fishing`, `LandPatches`, `OtherShip`, `SeaClutterWaves`, `SeaPatches`, `StrongFalseAlarms`, and `Tanker`. Six scene classes are derived from `MarineCategory`, and four ship classes are built from filtered subsets of `ShipCategory`. The official raw package contains exact-content duplicate files; these duplicates do not contribute additional samples and should be removed during preprocessing to obtain a clean benchmark. An exact byte-for-byte reproduction of the SAR-JEPA split cannot be recovered from the paper description alone. Reproducing that split exactly would require either an explicit split manifest or an officially released benchmark copy.

The raw directory mapping used to construct the benchmark is summarized below.

| Benchmark class       | Official raw source                  |
| --------------------- | ------------------------------------ |
| `Bridges`             | `MarineCategory/bridge`              |
| `CoastalLands_island` | `MarineCategory/landseaside_island`  |
| `LandPatches`         | `MarineCategory/land`                |
| `SeaClutterWaves`     | `MarineCategory/wave`                |
| `SeaPatches`          | `MarineCategory/sea`                 |
| `StrongFalseAlarms`   | `MarineCategory/likeship`            |
| `Cargo`               | `ShipCategory/Cargo/CargoShip`       |
| `Fishing`             | `ShipCategory/Fishing/Fishing`       |
| `OtherShip`           | `ShipCategory/Other/Othertypeofship` |
| `Tanker`\*            | `ShipCategory/Tanker`                |

`Tanker` is not constructed from the full raw `ShipCategory/Tanker` directory. Instead, it corresponds to a benchmark subset, likely excluding very small tanker subcategories that are difficult to distribute across training and validation splits.

**Statistics**

<table>
  <thead>
    <tr>
      <th>Split</th>
      <th>Class</th>
      <th>Count</th>
      <th>Class</th>
      <th>Count</th>
      <th>Class</th>
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">train</td>
      <td>Bridges</td>
      <td>1023</td>
      <td>Cargo</td>
      <td>366</td>
      <td>CoastalLands_island</td>
      <td>707</td>
    </tr>
    <tr>
      <td>Fishing</td>
      <td>248</td>
      <td>LandPatches</td>
      <td>1137</td>
      <td>OtherShip</td>
      <td>312</td>
    </tr>
    <tr>
      <td>SeaClutterWaves</td>
      <td>1378</td>
      <td>SeaPatches</td>
      <td>1250</td>
      <td>StrongFalseAlarms</td>
      <td>299</td>
    </tr>
    <tr>
      <td>Tanker</td>
      <td>150</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
    <tr>
      <td rowspan="4">test</td>
      <td>Bridges</td>
      <td>438</td>
      <td>Cargo</td>
      <td>156</td>
      <td>CoastalLands_island</td>
      <td>303</td>
    </tr>
    <tr>
      <td>Fishing</td>
      <td>106</td>
      <td>LandPatches</td>
      <td>487</td>
      <td>OtherShip</td>
      <td>133</td>
    </tr>
    <tr>
      <td>SeaClutterWaves</td>
      <td>590</td>
      <td>SeaPatches</td>
      <td>535</td>
      <td>StrongFalseAlarms</td>
      <td>128</td>
    </tr>
    <tr>
      <td>Tanker</td>
      <td>64</td>
      <td></td>
      <td></td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>

#### SOC40 (ATRNet-STAR)

`SOC40` uses the `SOC_40classes` subset of the public ATRNet-STAR release
(`Ground_Range/Amplitude_8bit`). Download the official release and point the
evaluation commands directly at that directory; it already follows the
`train/<class>` and `test/<class>` ImageFolder layout (see the `SOC40` path in
`docs/eval_guide.md`).

#### SAR-ACD

To prepare `SAR-ACD`, download the official release and place the class folders directly under `experiments/datasets/evaluation/SAR-ACD`. The expected directory structure is:

```text
<class>/*
```

We keep the original six-class release as the default benchmark, with classes `A220`, `A320321`, `A330`, `ARJ21`, `Boeing737`, and `Boeing787`. A SAR-JEPA-compatible variant can be obtained by removing `A320321`, but this repository uses the original six-class version by default. For random-shot evaluation, the ImageFolder-style root directory is sufficient. Exact reproduction of a published few-shot result still depends on the split definition and sampling strategy implemented in the evaluation code.

**Statistics**

| Class | Count | Class     | Count | Class     | Count |
| ----- | ----- | --------- | ----- | --------- | ----- |
| A220  | 464   | A320321   | 510   | A330      | 512   |
| ARJ21 | 514   | Boeing737 | 528   | Boeing787 | 504   |

### Detection

This subsection is reserved for future evaluation datasets for detection tasks. Each dataset entry should document the official source, annotation format, preprocessing and conversion pipeline, split protocol, output directory layout, and summary statistics needed for reproducible evaluation.

### Segmentation

This subsection is reserved for future evaluation datasets for segmentation tasks. Each dataset entry should document the official source, mask or polygon format, preprocessing and conversion pipeline, split protocol, output directory layout, and summary statistics needed for reproducible evaluation.

## References

1. Ross, T. D., Worrell, S. W., Velten, V. J., Mossing, J. C., and Bryant, M. L. "Standard SAR ATR Evaluation Experiments Using the MSTAR Public Release Data Set." _Proceedings of SPIE_, 1998. https://doi.org/10.1117/12.321859
2. Hou, X., Ao, W., Song, Q., Lai, J., Wang, H., and Xu, F. "FUSAR-Ship: Building a High-Resolution SAR-AIS Matchup Dataset of Gaofen-3 for Ship Detection and Recognition." _Science China Information Sciences_, 2020. https://doi.org/10.1007/s11432-019-2772-5
3. Wang, D., Song, Y., Huang, J., An, D., and Chen, L. "SAR Target Classification Based on Multiscale Attention Super-Class Network." _IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing_, 15: 9004-9019, 2022. https://doi.org/10.1109/JSTARS.2022.3206901
4. Sun, X., Lv, Y., Wang, Z., and Fu, K. "SCAN: Scattering Characteristics Analysis Network for Few-Shot Aircraft Classification in High-Resolution SAR Images." _IEEE Transactions on Geoscience and Remote Sensing_, 60: 1-17, 2022. https://doi.org/10.1109/TGRS.2022.3166174
5. Li, W., Yang, W., Liu, T., Hou, Y., Li, Y., Liu, Z., Liu, Y., and Liu, L. "Predicting Gradient Is Better: Exploring Self-Supervised Learning for SAR ATR with a Joint-Embedding Predictive Architecture." _ISPRS Journal of Photogrammetry and Remote Sensing_, 218: 326-338, 2024. https://doi.org/10.1016/j.isprsjprs.2024.09.013
