# Mars Dataset Tools

The detector pipeline consumes an external prepared dataset; it does not require the raw-data download step at runtime.

## Data sources

THEMIS imagery is publicly available through the [THEMIS Image Explorer](http://viewer.mars.asu.edu/faq) and NASA's [Planetary Data System](http://pds-imaging.jpl.nasa.gov/). The crater metadata used by the historical preparation scripts came from the referenced Mars crater study dataset. Download and store those external assets outside this repository.

## Expected prepared layout

The later CutLER registration expects a dataset root with this structure:

```text
<dataset-root>/
├── train/
├── val/
├── test/
└── annotations/
    ├── <initial-annotations>.json
    ├── train_r1.json
    ├── train_r2.json
    └── train_r3.json
```

The image list supplied to `scripts/run_label_generation.py` is relative to the Grounded SAM dataset root. The resulting filtered COCO file should be used as the initial annotation file for training.

Generated imagery, downloaded CSV files, caches, and outputs are not repository contents.
