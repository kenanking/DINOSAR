# CAMC manuscript figures

`input.webp`, `content.webp`, and `anchors.webp` are panels extracted from
`figures/content_aware_multicrop.eps` in the DINOSAR manuscript. Ghostscript
rendered the EPS at 120 dpi; the extracts are lossless WebP.

`examples.svg` is the original author-supplied `图片1.svg`, preserved byte for
byte. Its embedded images and vector annotations are unchanged. The HTML uses
SVG viewports to present the three columns separately and reflow them on mobile;
all nine panels share this single source. Panels are displayed inline without zoom interactions.

The HTML workflow follows “SAR Content Prior”, “Content-Aware Multi-Crop”, and
Algorithm 1. Its global-layout SVGs are conceptual schematics. The qualitative
examples retain the manuscript's solid global / dashed local convention; the
separate motivation experiment uses its own explicitly stated convention.
