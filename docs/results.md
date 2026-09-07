# Experimental results

[Back to the project overview](../README.md)

Tables reproduce the manuscript’s rows, numerical precision, and emphasis.
**Bold** and <ins>underlined</ins> values indicate best and second-best results,
respectively, within the comparison scope stated for each table. SAR-1M rows
are separate pretraining controls; the released checkpoints use UniSAR-7M.


### Frozen-backbone classification

Top-1 accuracy (%); LP denotes linear probing.

<table>
<thead>
<tr><th scope="col" rowspan="2">Method</th><th scope="col" rowspan="2">Params (M)</th><th scope="col" colspan="2">MSTAR SOC-10</th><th scope="col" colspan="2">FUSAR-Ship</th><th scope="col" colspan="2">ATRNet-STAR SOC-40</th></tr>
<tr><th scope="col"><i>k</i>-NN</th><th scope="col">LP</th><th scope="col"><i>k</i>-NN</th><th scope="col">LP</th><th scope="col"><i>k</i>-NN</th><th scope="col">LP</th></tr>
</thead>
<tbody>
<tr><th scope="row" align="left"><em>SAR-JEPA</em></th><td align="center">85.3</td><td align="center">94.10</td><td align="center">97.57</td><td align="center">93.50</td><td align="center">94.32</td><td align="center">43.84</td><td align="center">38.55</td></tr>
<tr><th scope="row" align="left"><em>SARATR-X</em></th><td align="center">65.8</td><td align="center"><ins>98.14</ins></td><td align="center"><strong>99.46</strong></td><td align="center">93.91</td><td align="center">94.80</td><td align="center">53.24</td><td align="center">44.12</td></tr>
<tr><th scope="row" align="left"><em>SUMMIT</em></th><td align="center">85.8</td><td align="center">86.80</td><td align="center">94.06</td><td align="center">92.86</td><td align="center">94.35</td><td align="center">43.74</td><td align="center">42.48</td></tr>
<tr><th scope="row" align="left"><em>SARMAE</em></th><td align="center">85.8</td><td align="center">90.72</td><td align="center">93.94</td><td align="center">93.47</td><td align="center">95.10</td><td align="center">46.92</td><td align="center">45.25</td></tr>
<tr><th scope="row" align="left"><strong>DINOSAR-B/16 (SAR-1M)</strong></th><td align="center">85.3</td><td align="center"><strong>98.31</strong></td><td align="center">98.80</td><td align="center"><ins>95.24</ins></td><td align="center"><ins>95.51</ins></td><td align="center"><ins>71.75</ins></td><td align="center"><ins>47.58</ins></td></tr>
<tr><th scope="row" align="left"><strong>DINOSAR-S/16 (UniSAR-7M)</strong></th><td align="center">21.4</td><td align="center">91.63</td><td align="center">97.20</td><td align="center">94.83</td><td align="center">93.88</td><td align="center">67.14</td><td align="center">44.81</td></tr>
<tr><th scope="row" align="left"><strong>DINOSAR-B/16 (UniSAR-7M)</strong></th><td align="center">85.3</td><td align="center">96.91</td><td align="center"><ins>98.97</ins></td><td align="center"><strong>95.41</strong></td><td align="center"><strong>95.58</strong></td><td align="center"><strong>77.02</strong></td><td align="center"><strong>60.79</strong></td></tr>
</tbody>
</table>


### Few-shot classification

Top-1 accuracy (%), mean ± standard deviation over five seeds.

<table>
<thead>
<tr><th scope="col" rowspan="2">Method</th><th scope="col" colspan="3">MSTAR SOC-10</th><th scope="col" colspan="3">FUSAR-Ship</th><th scope="col" colspan="3">SAR-ACD</th></tr>
<tr><th scope="col">10-shot</th><th scope="col">20-shot</th><th scope="col">40-shot</th><th scope="col">10-shot</th><th scope="col">20-shot</th><th scope="col">40-shot</th><th scope="col">10-shot</th><th scope="col">20-shot</th><th scope="col">40-shot</th></tr>
</thead>
<tbody>
<tr><th scope="row" align="left"><em>SAR-JEPA</em></th><td align="center">79.8 ± 1.2</td><td align="center">89.3 ± 1.1</td><td align="center">94.3 ± 0.5</td><td align="center">82.8 ± 0.8</td><td align="center">87.4 ± 0.7</td><td align="center">89.6 ± 0.4</td><td align="center">40.5 ± 4.1</td><td align="center">49.7 ± 1.5</td><td align="center">58.3 ± 1.1</td></tr>
<tr><th scope="row" align="left"><em>SARATR-X</em></th><td align="center"><strong>95.6 ± 0.5</strong></td><td align="center"><strong>97.9 ± 0.5</strong></td><td align="center"><strong>99.1 ± 0.3</strong></td><td align="center">83.1 ± 1.0</td><td align="center">86.4 ± 1.0</td><td align="center">89.0 ± 1.1</td><td align="center">48.6 ± 2.4</td><td align="center">56.0 ± 0.9</td><td align="center">66.3 ± 0.8</td></tr>
<tr><th scope="row" align="left"><em>SUMMIT</em></th><td align="center">78.6 ± 2.3</td><td align="center">84.1 ± 0.8</td><td align="center">87.8 ± 1.7</td><td align="center">82.3 ± 1.4</td><td align="center">86.1 ± 1.5</td><td align="center">89.5 ± 0.5</td><td align="center">38.1 ± 3.1</td><td align="center">49.0 ± 1.6</td><td align="center">61.6 ± 2.4</td></tr>
<tr><th scope="row" align="left"><em>SARMAE</em></th><td align="center">83.2 ± 2.4</td><td align="center">87.5 ± 0.9</td><td align="center">89.0 ± 1.7</td><td align="center">82.5 ± 1.0</td><td align="center">85.9 ± 0.8</td><td align="center">89.7 ± 0.9</td><td align="center">50.0 ± 4.1</td><td align="center">56.7 ± 1.8</td><td align="center">69.3 ± 2.3</td></tr>
<tr><th scope="row" align="left"><strong>DINOSAR-B/16 (SAR-1M)</strong></th><td align="center">88.4 ± 1.6</td><td align="center"><ins>93.6 ± 0.5</ins></td><td align="center"><ins>97.2 ± 0.4</ins></td><td align="center"><strong>89.3 ± 1.2</strong></td><td align="center"><strong>91.2 ± 0.5</strong></td><td align="center"><strong>92.3 ± 0.4</strong></td><td align="center"><ins>55.2 ± 2.9</ins></td><td align="center"><ins>63.2 ± 1.4</ins></td><td align="center"><ins>75.3 ± 1.0</ins></td></tr>
<tr><th scope="row" align="left"><strong>DINOSAR-S/16 (UniSAR-7M)</strong></th><td align="center">89.7 ± 1.4</td><td align="center">92.8 ± 1.1</td><td align="center">96.3 ± 0.2</td><td align="center">88.3 ± 0.6</td><td align="center"><ins>90.6 ± 1.3</ins></td><td align="center"><ins>92.1 ± 0.3</ins></td><td align="center">54.9 ± 2.5</td><td align="center">63.1 ± 1.7</td><td align="center">74.1 ± 0.8</td></tr>
<tr><th scope="row" align="left"><strong>DINOSAR-B/16 (UniSAR-7M)</strong></th><td align="center"><ins>90.7 ± 0.8</ins></td><td align="center">93.3 ± 1.3</td><td align="center">97.0 ± 0.4</td><td align="center"><ins>88.5 ± 0.5</ins></td><td align="center">90.2 ± 0.9</td><td align="center"><ins>92.1 ± 0.3</ins></td><td align="center"><strong>58.0 ± 2.0</strong></td><td align="center"><strong>65.6 ± 2.1</strong></td><td align="center"><strong>76.2 ± 1.2</strong></td></tr>
</tbody>
</table>


### Full fine-tuning on ATRNet-STAR

Top-1 accuracy (%). † Results quoted from ATRBench.

<table>
<thead>
<tr><th scope="col">Method</th><th scope="col">SOC-40</th><th scope="col">SOC-50</th><th scope="col">EOC-Scene</th><th scope="col">EOC-Dep.</th><th scope="col">EOC-Az.</th><th scope="col">EOC-Band</th><th scope="col">EOC-Pol.</th></tr>
</thead>
<tbody>
<tr><th scope="row" align="left"><em>VGG16</em><sup>†</sup></th><td align="center">88.8</td><td align="center">72.9</td><td align="center">21.6</td><td align="center">33.2</td><td align="center">15.7</td><td align="center">78.8</td><td align="center">72.5</td></tr>
<tr><th scope="row" align="left"><em>ResNet-18</em><sup>†</sup></th><td align="center">90.6</td><td align="center">71.2</td><td align="center">16.1</td><td align="center">33.9</td><td align="center">14.9</td><td align="center">83.1</td><td align="center">71.4</td></tr>
<tr><th scope="row" align="left"><em>ResNet-34</em><sup>†</sup></th><td align="center">91.7</td><td align="center">72.9</td><td align="center">18.0</td><td align="center">37.2</td><td align="center">16.5</td><td align="center">83.8</td><td align="center">70.5</td></tr>
<tr><th scope="row" align="left"><em>ConvNeXt</em><sup>†</sup></th><td align="center">96.0</td><td align="center">81.6</td><td align="center">16.5</td><td align="center">43.1</td><td align="center">21.1</td><td align="center">88.4</td><td align="center">83.1</td></tr>
<tr><th scope="row" align="left"><em>ViT-B/16</em><sup>†</sup></th><td align="center">76.4</td><td align="center">59.2</td><td align="center">12.9</td><td align="center">30.4</td><td align="center">29.0</td><td align="center">65.7</td><td align="center">53.6</td></tr>
<tr><th scope="row" align="left"><em>HiViT-B</em><sup>†</sup></th><td align="center">86.8</td><td align="center">68.0</td><td align="center">15.8</td><td align="center">31.4</td><td align="center">22.8</td><td align="center">70.7</td><td align="center">67.1</td></tr>
<tr><th scope="row" align="left"><em>HDANet</em><sup>†</sup></th><td align="center">89.1</td><td align="center">63.7</td><td align="center"><strong>33.6</strong></td><td align="center">32.9</td><td align="center">16.4</td><td align="center">79.5</td><td align="center">63.1</td></tr>
<tr><th scope="row" align="left"><em>SARATR-X</em></th><td align="center">96.1</td><td align="center">79.4</td><td align="center">13.9</td><td align="center">37.7</td><td align="center">37.5</td><td align="center">86.4</td><td align="center">72.3</td></tr>
<tr><th scope="row" align="left"><em>SAR-JEPA</em></th><td align="center">91.7</td><td align="center">68.9</td><td align="center">12.6</td><td align="center">31.6</td><td align="center">33.9</td><td align="center">79.0</td><td align="center">60.6</td></tr>
<tr><th scope="row" align="left"><em>SUMMIT</em></th><td align="center">97.9</td><td align="center">85.8</td><td align="center">24.3</td><td align="center">44.5</td><td align="center">43.4</td><td align="center">91.9</td><td align="center">83.0</td></tr>
<tr><th scope="row" align="left"><em>SARMAE</em></th><td align="center">98.1</td><td align="center">86.5</td><td align="center">19.9</td><td align="center">44.0</td><td align="center">42.3</td><td align="center">92.6</td><td align="center">84.0</td></tr>
<tr><th scope="row" align="left"><strong>DINOSAR-B/16 (SAR-1M)</strong></th><td align="center">98.0</td><td align="center">87.4</td><td align="center">16.0</td><td align="center">43.2</td><td align="center">42.9</td><td align="center">92.3</td><td align="center">81.1</td></tr>
<tr><th scope="row" align="left"><strong>DINOSAR-S/16 (UniSAR-7M)</strong></th><td align="center"><ins>98.8</ins></td><td align="center"><ins>90.3</ins></td><td align="center">21.0</td><td align="center"><ins>46.9</ins></td><td align="center"><ins>46.7</ins></td><td align="center"><ins>95.7</ins></td><td align="center"><ins>87.8</ins></td></tr>
<tr><th scope="row" align="left"><strong>DINOSAR-B/16 (UniSAR-7M)</strong></th><td align="center"><strong>99.1</strong></td><td align="center"><strong>92.4</strong></td><td align="center"><ins>24.5</ins></td><td align="center"><strong>50.8</strong></td><td align="center"><strong>49.6</strong></td><td align="center"><strong>96.9</strong></td><td align="center"><strong>90.8</strong></td></tr>
</tbody>
</table>


### Object detection: SARDet-100K, 1× schedule

COCO-style AP. Subscripts 50/75 denote IoU thresholds; S/M/L denote object sizes. † Results quoted from the SARDet-100K paper.

<table>
<thead>
<tr><th scope="col">Method</th><th scope="col">AP</th><th scope="col">AP<sub>50</sub></th><th scope="col">AP<sub>75</sub></th><th scope="col">AP<sub>S</sub></th><th scope="col">AP<sub>M</sub></th><th scope="col">AP<sub>L</sub></th></tr>
</thead>
<tbody>
<tr><th scope="row" align="left"><em>Faster R-CNN</em><sup>†</sup></th><td align="center">49.0</td><td align="center">82.2</td><td align="center">52.9</td><td align="center">43.5</td><td align="center">60.6</td><td align="center">55.0</td></tr>
<tr><th scope="row" align="left"><em>Cascade R-CNN</em><sup>†</sup></th><td align="center">51.1</td><td align="center">81.9</td><td align="center">55.8</td><td align="center">44.9</td><td align="center">62.9</td><td align="center">60.3</td></tr>
<tr><th scope="row" align="left"><em>Deformable DETR</em><sup>†</sup></th><td align="center">50.0</td><td align="center">85.1</td><td align="center">51.7</td><td align="center">44.0</td><td align="center">65.1</td><td align="center">61.2</td></tr>
<tr><th scope="row" align="left"><em>ConvNeXt-B</em><sup>†</sup></th><td align="center"><ins>55.1</ins></td><td align="center">87.8</td><td align="center">59.5</td><td align="center">48.9</td><td align="center"><strong>66.9</strong></td><td align="center">61.1</td></tr>
<tr><th scope="row" align="left"><em>Swin-B</em><sup>†</sup></th><td align="center">53.8</td><td align="center">87.8</td><td align="center">59.0</td><td align="center">49.1</td><td align="center">64.6</td><td align="center">60.0</td></tr>
<tr><th scope="row" align="left"><em>SARATR-X</em></th><td align="center">54.20</td><td align="center">88.09</td><td align="center">58.14</td><td align="center">48.74</td><td align="center">65.44</td><td align="center"><ins>63.76</ins></td></tr>
<tr><th scope="row" align="left"><em>SUMMIT</em></th><td align="center">50.86</td><td align="center">85.88</td><td align="center">54.33</td><td align="center">45.04</td><td align="center">61.73</td><td align="center">60.14</td></tr>
<tr><th scope="row" align="left"><em>SARMAE</em></th><td align="center">52.00</td><td align="center">86.26</td><td align="center">55.97</td><td align="center">46.69</td><td align="center">62.60</td><td align="center">60.67</td></tr>
<tr><th scope="row" align="left"><strong>DINOSAR-B/16 (SAR-1M)</strong></th><td align="center">52.52</td><td align="center">87.39</td><td align="center">56.77</td><td align="center">47.54</td><td align="center">63.03</td><td align="center">60.38</td></tr>
<tr><th scope="row" align="left"><strong>Random-crop DINO (UniSAR-7M)</strong></th><td align="center">54.22</td><td align="center"><ins>88.43</ins></td><td align="center"><ins>59.55</ins></td><td align="center"><ins>49.37</ins></td><td align="center">64.99</td><td align="center">62.47</td></tr>
<tr><th scope="row" align="left"><strong>DINOSAR-B/16 (UniSAR-7M)</strong></th><td align="center"><strong>55.28</strong></td><td align="center"><strong>89.08</strong></td><td align="center"><strong>61.05</strong></td><td align="center"><strong>49.52</strong></td><td align="center"><ins>66.85</ins></td><td align="center"><strong>64.64</strong></td></tr>
</tbody>
</table>


### Object detection: SARDet-100K, 3× schedule

COCO-style AP. Best/second-best markings apply only to the unified-protocol rows. † Originally reported SARATR-X result uses multi-scale 480–800 training and 800-pixel short-side testing; it is not directly comparable.

<table>
<thead>
<tr><th scope="col">Method</th><th scope="col">AP</th><th scope="col">AP<sub>50</sub></th><th scope="col">AP<sub>75</sub></th><th scope="col">AP<sub>S</sub></th><th scope="col">AP<sub>M</sub></th><th scope="col">AP<sub>L</sub></th></tr>
</thead>
<tbody>
<tr><th scope="row" align="left"><em>SARATR-X</em><sup>†</sup></th><td align="center">57.3</td><td align="center">88.7</td><td align="center">62.8</td><td align="center">—</td><td align="center">—</td><td align="center">—</td></tr>
<tr><th scope="row" align="left"><em>SARATR-X</em></th><td align="center"><ins>59.76</ins></td><td align="center"><ins>89.81</ins></td><td align="center"><ins>64.21</ins></td><td align="center"><ins>56.63</ins></td><td align="center"><ins>71.20</ins></td><td align="center"><ins>66.97</ins></td></tr>
<tr><th scope="row" align="left"><em>SARMAE</em></th><td align="center">56.88</td><td align="center">87.24</td><td align="center">60.22</td><td align="center">54.36</td><td align="center">68.09</td><td align="center">62.61</td></tr>
<tr><th scope="row" align="left"><em>SUMMIT</em></th><td align="center">56.56</td><td align="center">87.61</td><td align="center">60.40</td><td align="center">53.08</td><td align="center">68.94</td><td align="center">64.35</td></tr>
<tr><th scope="row" align="left"><strong>DINOSAR-B/16 (UniSAR-7M)</strong></th><td align="center"><strong>61.04</strong></td><td align="center"><strong>90.89</strong></td><td align="center"><strong>64.97</strong></td><td align="center"><strong>58.20</strong></td><td align="center"><strong>72.47</strong></td><td align="center"><strong>69.81</strong></td></tr>
</tbody>
</table>


### Semantic segmentation: AIR-PolSAR-Seg v1.0

UPerNet; mean over three independent runs. Metrics are mIoU and per-class IoU (%). No v1.0 training or test sample is included in UniSAR-7M.

<table>
<thead>
<tr><th scope="col">Method</th><th scope="col">mIoU</th><th scope="col">Industrial</th><th scope="col">Natural</th><th scope="col">Water</th><th scope="col">Land Use</th><th scope="col">Housing</th><th scope="col">Other</th></tr>
</thead>
<tbody>
<tr><th scope="row" align="left"><em>SAR-JEPA</em></th><td align="center">49.37</td><td align="center">34.15</td><td align="center">72.81</td><td align="center">72.70</td><td align="center">0.00</td><td align="center">58.70</td><td align="center">57.87</td></tr>
<tr><th scope="row" align="left"><em>SARATR-X</em></th><td align="center">52.80</td><td align="center">39.36</td><td align="center">74.40</td><td align="center">75.24</td><td align="center">1.51</td><td align="center">66.08</td><td align="center">60.21</td></tr>
<tr><th scope="row" align="left"><em>SARMAE</em></th><td align="center">56.96</td><td align="center">43.41</td><td align="center">74.68</td><td align="center">75.55</td><td align="center"><strong>13.15</strong></td><td align="center">65.72</td><td align="center">69.24</td></tr>
<tr><th scope="row" align="left"><em>SUMMIT</em></th><td align="center"><strong>59.31</strong></td><td align="center"><strong>50.81</strong></td><td align="center"><strong>78.11</strong></td><td align="center"><ins>77.37</ins></td><td align="center"><ins>10.45</ins></td><td align="center"><strong>72.15</strong></td><td align="center">66.96</td></tr>
<tr><th scope="row" align="left"><strong>DINOSAR-B/16 (SAR-1M)</strong></th><td align="center">57.24</td><td align="center">42.62</td><td align="center">74.97</td><td align="center">76.87</td><td align="center">8.60</td><td align="center">68.55</td><td align="center"><ins>71.81</ins></td></tr>
<tr><th scope="row" align="left"><strong>DINOSAR-B/16 (UniSAR-7M)</strong></th><td align="center"><ins>58.20</ins></td><td align="center"><ins>47.07</ins></td><td align="center"><ins>77.10</ins></td><td align="center"><strong>80.15</strong></td><td align="center">1.23</td><td align="center"><ins>69.29</ins></td><td align="center"><strong>74.33</strong></td></tr>
</tbody>
</table>


### Cross-modal retrieval: SARVLM

Recall (%), official 5,000-pair evaluation set. MeanRecall averages all six recall values. † SARCLIP results are quoted from the SARVLM paper; these LAION-initialized reference models are excluded from best/second-best marking.

<table>
<thead>
<tr><th scope="col" rowspan="2">Method</th><th scope="col" colspan="3">Image → Text</th><th scope="col" colspan="3">Text → Image</th><th scope="col" rowspan="2">MeanRecall</th></tr>
<tr><th scope="col">R@1</th><th scope="col">R@5</th><th scope="col">R@10</th><th scope="col">R@1</th><th scope="col">R@5</th><th scope="col">R@10</th></tr>
</thead>
<tbody>
<tr><th scope="row" align="left"><em>SARCLIP ViT-B/32</em><sup>†</sup></th><td align="center">10.20</td><td align="center">28.54</td><td align="center">41.26</td><td align="center">10.84</td><td align="center">29.86</td><td align="center">42.18</td><td align="center">27.15</td></tr>
<tr><th scope="row" align="left"><em>SARCLIP ViT-L/14</em><sup>†</sup></th><td align="center">11.84</td><td align="center">32.60</td><td align="center">44.04</td><td align="center">12.88</td><td align="center">33.72</td><td align="center">44.68</td><td align="center">29.96</td></tr>
<tr><th scope="row" align="left"><em>SARATR-X</em></th><td align="center">8.28</td><td align="center">26.06</td><td align="center">38.14</td><td align="center">9.06</td><td align="center">27.78</td><td align="center">38.46</td><td align="center">24.63</td></tr>
<tr><th scope="row" align="left"><em>SARMAE</em></th><td align="center"><ins>9.22</ins></td><td align="center">27.20</td><td align="center">38.98</td><td align="center"><ins>9.72</ins></td><td align="center">28.40</td><td align="center">40.22</td><td align="center">25.62</td></tr>
<tr><th scope="row" align="left"><em>SUMMIT</em></th><td align="center"><ins>9.22</ins></td><td align="center"><ins>28.72</ins></td><td align="center"><strong>40.92</strong></td><td align="center">9.68</td><td align="center"><ins>29.42</ins></td><td align="center"><ins>41.52</ins></td><td align="center"><ins>26.58</ins></td></tr>
<tr><th scope="row" align="left"><strong>DINOSAR-B/16</strong></th><td align="center"><strong>10.08</strong></td><td align="center"><strong>28.76</strong></td><td align="center"><ins>40.86</ins></td><td align="center"><strong>10.78</strong></td><td align="center"><strong>30.88</strong></td><td align="center"><strong>41.64</strong></td><td align="center"><strong>27.17</strong></td></tr>
</tbody>
</table>


## Ablation studies


### CAMC component ablations

ViT-S/16, UniSAR-7M, 5 pretraining epochs, effective batch size 1280; frozen evaluation on ATRNet-STAR SOC-40. Top-1 accuracy (%), mean ± standard deviation over three runs. Bold denotes the best mean. All CAMC variants share content-aware proposal and global-view construction.

<table>
<thead>
<tr><th scope="col">Variant</th><th scope="col">k-NN k=1</th><th scope="col">k-NN k=5</th><th scope="col">LP</th></tr>
</thead>
<tbody>
<tr><th scope="row" align="left">Random-crop DINO</th><td align="center">11.78 ± 0.52</td><td align="center">13.30 ± 0.53</td><td align="center">19.87 ± 0.22</td></tr>
<tr><th scope="row" align="left">w/o content term (T̄≡ 1)</th><td align="center">15.13 ± 0.99</td><td align="center">17.07 ± 0.98</td><td align="center">21.85 ± 0.18</td></tr>
<tr><th scope="row" align="left">w/o anchor proximity (A≡ 1)</th><td align="center">17.83 ± 0.57</td><td align="center">20.02 ± 0.43</td><td align="center">22.77 ± 0.15</td></tr>
<tr><th scope="row" align="left">w/o coverage term (U≡ 1)</th><td align="center">17.53 ± 1.60</td><td align="center">19.78 ± 1.62</td><td align="center">22.34 ± 0.36</td></tr>
<tr><th scope="row" align="left">Fixed concentration c=2/3</th><td align="center">18.13 ± 0.76</td><td align="center">20.40 ± 0.73</td><td align="center">22.59 ± 0.31</td></tr>
<tr><th scope="row" align="left">Full CAMC (<strong>ours</strong>)</th><td align="center"><strong>19.76</strong> ± 0.64</td><td align="center"><strong>21.98</strong> ± 0.85</td><td align="center"><strong>22.84</strong> ± 0.38</td></tr>
</tbody>
</table>
