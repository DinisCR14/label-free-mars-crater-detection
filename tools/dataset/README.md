# Mars Dataset Tools

The detector pipeline consumes an external prepared dataset; it does not require the raw-data download step at runtime.

## Data sources

THEMIS imagery is publicly available through the [THEMIS Image Explorer](http://viewer.mars.asu.edu/faq) and NASA's [Planetary Data System](http://pds-imaging.jpl.nasa.gov/). The crater metadata used by the historical preparation scripts came from the referenced Mars crater study dataset. Download and store those external assets outside this repository.

The published experiment used a prepared 512x512 archive supplied by the project supervisors: [Google Drive archive](https://drive.google.com/file/d/1DcEOBtmu6AMFUOjjpm2qBpcfyQGBBawa/view?usp=sharing). It contains `_original.png` images, matching `_marked.png` visualizations, and `labels.json`. The images were created from THEMIS imagery. This repository does not redistribute that archive. The helper download script retrieves raw THEMIS source imagery only; it does not reproduce the supplied 512x512 archive.

The historical preparation material also used the [Mars Crater Study Dataset](https://www.kaggle.com/datasets/codebreaker619/mars-crater-study-dataset/) for crater locations. Keep downloaded archives and source data outside the repository.

The supplied `regularhd-large/labels.json` was compared against the 2012, 2014, and 2020 Robbins and Hynek catalogue files. Its geographic coordinates match the 2012 catalogue at floating-point precision, and its recovered diameters match to the catalogue's rounding precision, identifying the 2012 catalogue as the source version currently in use. The comparison tool is [compare_catalog_versions.py](compare_catalog_versions.py). Keep the catalogue files and generated `catalogue_comparison_report.json` in the external `/data/mars/robbins2020-work/catalogues/` directory. The 2014 and 2020 versions are preserved there for future analysis but are not currently used.

Reproduce the comparison with:

```bash
python tools/dataset/compare_catalog_versions.py \
    /data/mars/regularhd-large/labels.json \
    /data/mars/robbins2020-work/catalogues/RobbinsCraters_20121016.tsv \
    "/data/mars/robbins2020-work/catalogues/Combined Catalog - 20140830 (v2).txt" \
    /data/mars/robbins2020-work/catalogues/Catalog_Mars_Release_2020_1kmPlus_FullMorphData.csv \
    /data/mars/robbins2020-work/catalogues/catalogue_comparison_report.json
```

## Corrected 2012 baseline

The supplied `regularhd-large` archive is the trusted 2012 image/label pair. The correction pipeline is run with the scripts in `tools/dataset/` and writes a separate output directory; it never modifies the source archive. The source archive and all generated datasets should remain outside this repository.

First audit the source-image coverage needed by the remosaicking step:

```bash
python tools/dataset/audit_correction_coverage.py \
    /data/mars/regularhd-large \
    /data/mars/robbins2020-work/correction_coverage_manifest.json
```

Then generate corrected images, overlays, correction metadata, and unfiltered transformed labels:

```bash
python tools/dataset/correct_reference_dataset.py \
    /data/mars/regularhd-large/labels.json \
    /data/mars/regularhd-large \
    /data/mars/regularhd-large-corrected-2012 \
    --coverage-manifest /data/mars/robbins2020-work/correction_coverage_manifest.json \
    --all
```

Finally create the canonical labels by removing only fully invisible ellipses and conservative within-tile duplicates:

```bash
python tools/dataset/filter_corrected_labels.py \
    /data/mars/regularhd-large-corrected-2012/labels_unfiltered.json \
    /data/mars/regularhd-large-corrected-2012/labels.json \
    /data/mars/regularhd-large-corrected-2012/label_filter_report.json
```

Build the final per-tile manifest from the three reports:

```bash
python tools/dataset/build_corrected_manifest.py \
    /data/mars/regularhd-large-corrected-2012/correction_report.json \
    /data/mars/regularhd-large-corrected-2012/label_filter_report.json \
    /data/mars/robbins2020-work/correction_coverage_manifest.json \
    /data/mars/regularhd-large-corrected-2012/dataset_manifest.json
```

The corrected images are latitude-row mosaics compressed horizontally by `cos(abs(latitude))`. Consequently, a corrected tile can contain labels from neighboring source tiles and the same crater can occur in multiple overlapping corrected tiles. The canonical `labels.json` keeps ellipses that intersect the output canvas, including ellipses whose centers are outside the canvas, and removes only fully invisible ellipses and conservative within-tile duplicates. The correction report records complete versus gapped source neighborhoods; gapped tiles contain black regions where source imagery is unavailable.

The final external dataset contains `labels.json`, `labels_unfiltered.json`, `correction_report.json`, `label_filter_report.json`, `dataset_manifest.json`, and corrected overlay previews. The validated run predates this standard naming and uses `labels_visible_deduplicated_report.json` for the filter report; it is otherwise equivalent. Keep this generated dataset outside the repository. The canonical labels from the validated run contain 485,267 ellipses; the unfiltered correction report contains 1,152,607 transformed labels before visibility filtering and deduplication.

The canonical correction path consists of `audit_correction_coverage.py`, `correct_reference_dataset.py`, `filter_corrected_labels.py`, and `build_corrected_manifest.py`. `audit_reference_labels.py` and `render_canonical_label_overlays.py` are optional validation tools. Catalog-comparison scripts, prototype scripts, and alternate overlay renderers are exploratory review tools kept in the external `robbins2020-work/exploration/` directory and are not required by dataset preparation.

## Expected prepared layout

The later CutLER registration expects a dataset root with this structure:

```text
<dataset-root>/
├── train/
├── val/
├── test/
├── marked/
├── labels.json
├── image-list.txt
├── split-manifest.json
└── annotations/
    ├── val_truegt.json
    ├── test_truegt_clipped.json
    ├── <initial-annotations>.json
    ├── train_r1.json
    ├── train_r2.json
    └── train_r3.json
```

The image list supplied to `scripts/run_label_generation.py` should list the training images, relative to the Grounded SAM dataset root. Set its `--image-root` argument to the directory containing those images, normally `<dataset-root>/train`. The resulting filtered COCO file should be used as the initial annotation file for training.

Prepare the downloaded supervisor archive with the repository script:

```bash
python tools/dataset/prepare_dataset.py \
    /data/mars-archive \
    /data/mars \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --max-abs-latitude 50 \
    --seed 0
```

The script keeps latitudes from -50 through 50 degrees inclusive, copies `_original.png` images into the three split directories, stores `_marked.png` visualizations under `marked/`, filters `labels.json` to the retained images, writes `split-manifest.json`, and generates `image-list.txt`. The seed makes the generated split repeatable; use the seed that matches the experiment if you have it.

Create the list from a flat training directory with paths relative to the dataset root:

```bash
find /data/mars/train -maxdepth 1 -type f \
    \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) \
    | sort \
    | sed 's#^/data/mars/##' \
    > /data/mars/image-list.txt
```

The resulting file contains one path per line, for example `train/tile_0001.png`. Keep only the original training images in this list; do not include marked visualizations or validation/test images.

## Ground-truth label conversion

The original preparation workflow stores crater labels in `labels.json` as a mapping from an image stem to axis-aligned ellipses:

```json
{"lat_-1.0_long_-20.0": [[128.0, 120.0, 18.0, 22.0]]}
```

Each ellipse is `[center_x, center_y, radius_x, radius_y]` in pixels and corresponds to an image named `<stem>_original.png`. Convert labels for each split with `crater_detection/export_coco.py`:

```bash
python crater_detection/export_coco.py \
    --ellipse-labels /data/mars/labels.json \
    --image-root /data/mars/val \
    --output-json /data/mars/annotations/val_truegt.json

python crater_detection/export_coco.py \
    --ellipse-labels /data/mars/labels.json \
    --image-root /data/mars/test \
    --output-json /data/mars/annotations/test_truegt_clipped.json \
    --clip-bboxes
```

The converter emits COCO polygon segmentations and bounding boxes. `--clip-bboxes` is intended for test ground truth so boxes are evaluated under the same in-image constraint as detector predictions. It does not alter the ellipse segmentation or the pseudo-label training workflow.

## Evaluate the final detector

After training, evaluate the final standard-loss checkpoint against the clipped test ground truth. Run this from the repository root and replace the checkpoint path if your output directory differs:

```bash
CRATER_DATASET_ROOT=/data/mars \
CRATER_ANNOTATIONS_DIR=/data/mars/annotations \
PYTHONPATH="$PWD/third_party/CutLER" \
python third_party/CutLER/cutler/train_net.py \
    --num-gpus 1 \
    --config-file third_party/CutLER/cutler/model_zoo/configs/CutLER-CraterDataset/cascade_mask_rcnn_R_50_FPN_final.yaml \
    --test-dataset crater_dataset_test \
    --eval-only \
    TEST.DETECTIONS_PER_IMAGE 100 \
    MODEL.WEIGHTS /data/mars-runs/training/final-standard-loss/model_final.pth \
    OUTPUT_DIR /data/mars-runs/evaluation
```

The evaluator reads `annotations/test_truegt_clipped.json` through the registered `crater_dataset_test` dataset and writes predictions and evaluation results under the selected output directory. `CRATER_ANNOTATIONS_DIR` is required because the annotation files are stored separately from the images.

Generated imagery, downloaded CSV files, caches, and outputs are not repository contents.
