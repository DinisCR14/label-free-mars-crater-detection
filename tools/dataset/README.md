# Mars Dataset Tools

The detector pipeline consumes an external prepared dataset; it does not require the raw-data download step at runtime.

## Canonical inference image manifest

Foundation-model inference and evaluation use an explicit image-to-ID
manifest. Build it once from the centralized corrected ground truth:

```bash
python tools/dataset/build_image_manifest.py \
  /home/dcortereal/datasets/ground-truth/corrected-centerlat/corrected-centerlat-clipped.json \
  /home/dcortereal/datasets/ground-truth/corrected-centerlat/corrected-centerlat-manifest.json
```

The manifest is external to the repository and owns the stable
`(id, file_name, width, height)` mapping. SAM 3, Grounding DINO, and Grounded
SAM must all receive it through `--manifest`. The runners reject missing or
extra corrected images, dimension mismatches, and stale prediction
checkpoints.

The primary bbox evaluation is implemented in
`tools/evaluation/evaluate_coco_bbox.py`. It uses `pycocotools.COCOeval`,
requires the clipped ground truth for image-bounded detector evaluation, and
reports standard AP/AR metrics plus the project's custom bbox-area size
categories (`pi*5^2` and `pi*20^2`). The older
`crater_detection/plot_bbox_metrics.py` script remains a supplementary custom
greedy-matching diagnostic and is not the authoritative evaluator. The
diagnostic implementation is now
`tools/evaluation/plot_bbox_diagnostics.py`; the old path remains a
compatibility wrapper.

Validate an output before evaluation:

```bash
python tools/dataset/validate_image_mapping.py \
  /home/dcortereal/datasets/ground-truth/corrected-centerlat/corrected-centerlat-manifest.json \
  /home/dcortereal/datasets/sam3-runs/corrected-centerlat.json
```

Use `--allow-legacy` only for existing outputs created before manifest
metadata was added; image records are still checked exactly.

## Data sources

THEMIS imagery is publicly available through the [THEMIS Image Explorer](http://viewer.mars.asu.edu/faq) and NASA's [Planetary Data System](http://pds-imaging.jpl.nasa.gov/). The crater metadata used by the historical preparation scripts came from the referenced Mars crater study dataset. Download and store those external assets outside this repository.

The published experiment used a prepared 512x512 archive supplied by the project supervisors: [Google Drive archive](https://drive.google.com/file/d/1DcEOBtmu6AMFUOjjpm2qBpcfyQGBBawa/view?usp=sharing). It contains `_original.png` images, matching `_marked.png` visualizations, and `labels.json`. The images were created from THEMIS imagery. This repository does not redistribute that archive. The helper download script retrieves raw THEMIS source imagery only; it does not reproduce the supplied 512x512 archive.

The historical preparation material also used the [Mars Crater Study Dataset](https://www.kaggle.com/datasets/codebreaker619/mars-crater-study-dataset/) for crater locations. Keep downloaded archives and source data outside the repository.

### Verified 2012 provenance checkpoint

An independent metadata-only reconstruction was started under `/data/mars/regularhd-really-2012/` using `RobbinsCraters_20121016.tsv` and the untouched `/data/mars/regularhd-large/` archive. The first five-tile sample established the following conventions:

- `lat_<latitude>_long_<longitude>` identifies the lower-left corner of a 2-degree tile.
- Pixel coordinates use 256 pixels per degree: `x = (longitude - tile_longitude) * 256` and `y = (latitude - tile_latitude) * 256`.
- Longitude differences must be normalized across the -180/180 boundary.
- Label radii are reconstructed from `DIAM_CIRCLE_IMAGE`, with the horizontal radius scaled by the cosine of the crater latitude.
- Existing labels match the 2012 catalogue at floating-point precision for the sampled tiles, including their recovered diameters.
- Boundary labels require an ellipse-intersection policy; centers may lie outside the nominal 512-by-512 tile.

The sample initially found boundary cases that required further classification. The full archive comparison is now complete: the historical preparation procedure has been identified in the associated `allexceptdata/tools/gen_dataset.py` and reproduced for the supplied 512x512 archive. The supplied `regularhd-large/labels.json` remains the canonical annotation; the repository generator is a provenance and reproducibility tool, not a replacement for that file.

The current `generate_corrected_labels.py` is not the 2012 reconstruction tool: its catalogue reader is specialized to the 2020 field names. The dedicated 2012 generator is [generate_robbins_2012_labels.py](generate_robbins_2012_labels.py).

### Historical label-filter reproduction

The associated `allexceptdata/tools/gen_dataset.py` contains the historical image-content filter. The repository generator reproduces it with `--historical-filter` while reading the existing archive images and writing only a separate labels file:

```bash
python tools/dataset/generate_robbins_2012_labels.py \
    /data/mars/catalogues/RobbinsCraters_20121016.tsv \
    /data/mars/regularhd-large \
    /data/mars/regularhd-reproduction/labels_historical_filter.json \
    /data/mars/regularhd-reproduction/labels_historical_filter_report.json \
    --historical-filter
```

This mode applies the catalogue minimum diameter of 0.25 km, the 8% axis-aligned bounding-box intersection threshold, the 8% non-black pixel threshold inside the clipped bounding box, and the 70% minimum non-black content threshold for a complete tile. It checks the historical first OpenCV channel, which corresponds to the blue channel when the archive PNG is read as RGB. It does not recreate or modify images.

The command reads the existing archive images as inputs and writes only `labels_historical_filter.json` and its report. It does not regenerate, copy, or modify images. The report records the catalogue path, catalogue SHA-256, image-directory path, tile geometry, filtering thresholds, and output counts. The paths in the example are portable placeholders; use the actual external paths on the machine running the command.

Against the current archive, the reproduction produced 390,699 labels across 12,264 tiles. The supplied `labels.json` contains 390,698 labels; 390,697 match after rounding coordinates to six decimals. The remaining differences are one supplied label at the exact 8% content boundary and two labels retained by the reproduction but absent from the supplied file. This identifies the historical cleaning policy, but those three occurrence-level differences remain an undocumented archive-generation detail and should not be silently presented as an exact reproduction.

The validated run used `/home/dcortereal/datasets/catalogues/RobbinsCraters_20121016.tsv` and `/home/dcortereal/datasets/regularhd-large`. The catalogue SHA-256 was `7dadc273277cd3a7c4155b2f92c67bda64a093b7d48abe19d6de2898198a27a6`; the full report is stored with the external reproduction outputs. Keep both external inputs read-only.

### Exploratory analyses

The following filters and visual audits were useful for understanding the discrepancy but are not part of the canonical 2012 reproduction command.

An optional exact-zero image-content filter removes a label only when at least one pixel center lies inside its projected ellipse and every covered pixel is grayscale zero. Applied to the intersection candidate, it removes 666 occurrences, including 665 candidate-only extras and one supplied occurrence at `lat_-4.0_long_80.0`; the filtered candidate contains 397,337 occurrences and still adds 6,640 occurrences beyond the supplied archive. Reproduce this analysis with `--discard-fully-black` and keep the resulting file separate until the historical inclusion policy is resolved.

A three-color visual audit of the filtered candidate found that 6,603 of its 6,640 candidate-only occurrences have centers outside their tile; the remaining 37 center-inside cases are mostly very large projected catalogue ellipses or labels adjacent to source-image gaps. The reproducible renderer is [render_label_comparison_overlays.py](render_label_comparison_overlays.py); use blue for shared labels, green for supplied-only labels, and red for candidate-only labels. These overlays are diagnostic only and are not required to generate the canonical reproduction.

An additional optional center-pixel filter was tested using the pixel at `floor(center_x), floor(center_y)`. It removed 426 occurrences from the fully-black-filtered candidate, but 402 of those occurrences are also present in the supplied archive; only 24 were candidate-only. The resulting candidate contains 396,911 occurrences, shares 390,295 with the supplied archive, and omits 403 supplied occurrences. Therefore a black center alone is not a reliable final-label rule and should remain an analysis option rather than part of the canonical policy.

The supplied `regularhd-large/labels.json` was compared against the 2012, 2014, and 2020 Robbins and Hynek catalogue files. Its geographic coordinates match the 2012 catalogue at floating-point precision, and its recovered diameters match to the catalogue's rounding precision, identifying the 2012 catalogue as the source version currently in use. The comparison tool is [compare_catalog_versions.py](compare_catalog_versions.py). Keep the catalogue files and generated `catalogue_comparison_report.json` in the external `/data/mars/catalogues/` directory. The 2014 and 2020 versions are preserved there for future analysis but are not currently used.

Reproduce the comparison with:

```bash
python tools/dataset/compare_catalog_versions.py \
    /data/mars/regularhd-large/labels.json \
    /data/mars/catalogues/RobbinsCraters_20121016.tsv \
    "/data/mars/catalogues/Combined Catalog - 20140830 (v2).txt" \
    /data/mars/catalogues/Catalog_Mars_Release_2020_1kmPlus_FullMorphData.csv \
    /data/mars/catalogues/catalogue_comparison_report.json
```

### Future work: using the 2014 or 2020 catalogue

The 2012, 2014, and 2020 catalogues must be treated as separate annotation
versions. The supplied `regularhd-large/labels.json` has been reproduced from
the 2012 catalogue and remains the current reference version. Do not replace
it with a later catalogue without first generating and validating a separate
label set.

There are two distinct transformations in this work:

1. **Image correction:** remosaicking neighboring source images and applying
    horizontal compression by `cos(abs(tile-center latitude))`. This corrects
    the image geometry and should be the same regardless of which catalogue is
    used.
2. **Catalogue-to-pixel projection:** converting a catalogue's geographic
    coordinates, diameters, and ellipse properties into source-image pixels.
    This may be catalogue-version-specific.

The 2014 changelog says that all image-based latitude, longitude, diameter, and
ellipse properties were updated using great-circle distances via Vincenty's
formulae, replacing the 2012 catalogue's simpler `cos(latitude)` projection.
The 2014 release also changed crater IDs, removed incomplete morphology and
depth fields below 3 km, and revised confidence and naming metadata. These
changes mean that differences from 2012 are not automatically evidence of
manual relabeling or improved crater boundaries.

The 2020 catalogue keeps most 2014 crater correspondences but applies an
automatic coordinate adjustment. It uses different column names, stores
longitudes in `0..360` rather than `-180..180`, and contains roughly 1,105
craters not matched to 2014. The accompanying catalogue review reports that
the direct 2012-versus-2020 analysis, latitude-dependent effects, and the choice
between circle and ellipse coordinates remain open questions.

To evaluate a later catalogue, create a separate catalogue-specific label
generation path. It should normalize longitude conventions, identify the
correct coordinate and ellipse fields, reproduce the relevant catalogue
projection, and then apply the common image correction. Compare the resulting
labels with the 2012 reference using numerical residuals and visual overlays,
especially for large center or diameter changes and high-latitude tiles. Keep
the 2014 or 2020 output as an alternative reference version until this review
is complete; do not use it as canonical ground truth solely because it is newer.

## Corrected 2012 baseline

The supplied `regularhd-large` archive is the trusted 2012 image/label pair. The correction pipeline is run with the scripts in `tools/dataset/` and writes a separate output directory; it never modifies the source archive. The source archive and all generated datasets should remain outside this repository.

The complete correction can be run with one command. It writes the coverage manifest and all other intermediate and final artifacts inside the output directory:

```bash
python tools/dataset/create_corrected_dataset.py \
    /data/mars/regularhd-large \
    /data/mars/regularhd-large-corrected-2012 \
    --labels /data/mars/regularhd-large/labels.json
```

The command runs the four stages below in order. To run them separately for debugging or partial reruns, use the individual scripts:

- `audit_correction_coverage.py`
- `correct_reference_dataset.py`
- `filter_corrected_labels.py`
- `build_corrected_manifest.py`

The corrected images are latitude-row mosaics compressed horizontally by `cos(abs(tile-center latitude))`, where the tile-center latitude is the filename latitude plus half the 2-degree tile extent. Consequently, a corrected tile can contain labels from neighboring source tiles and the same crater can occur in multiple overlapping corrected tiles. The canonical `labels.json` keeps ellipses that intersect the output canvas, including ellipses whose centers are outside the canvas, and removes only fully invisible ellipses and conservative within-tile duplicates. The correction report records complete versus gapped source neighborhoods; gapped tiles contain black regions where source imagery is unavailable.

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

For predictions that already contain a COCO `images` list, use `--image-metadata-json` to align label keys with prediction image IDs without reading the source images:

```bash
python crater_detection/export_coco.py \
    --ellipse-labels /data/mars/labels.json \
    --image-metadata-json /data/mars-runs/sam3/predictions.json \
    --output-json /data/mars-runs/sam3/ground_truth.json
```

For the corrected-centerlat experiment, keep detector-independent evaluation
files together under `/home/dcortereal/datasets/ground-truth/corrected-centerlat/`.
Generate the raw and clipped variants from the same image metadata:

```bash
python crater_detection/export_coco.py \
    --ellipse-labels /home/dcortereal/datasets/corrected-centerlat/labels.json \
    --image-metadata-json /home/dcortereal/datasets/sam3-runs/full-corrected-centerlat/sam3-circle-threshold-04.json \
    --output-json /home/dcortereal/datasets/ground-truth/corrected-centerlat/corrected-centerlat-raw.json

python crater_detection/export_coco.py \
    --ellipse-labels /home/dcortereal/datasets/corrected-centerlat/labels.json \
    --image-metadata-json /home/dcortereal/datasets/sam3-runs/full-corrected-centerlat/sam3-circle-threshold-04.json \
    --output-json /home/dcortereal/datasets/ground-truth/corrected-centerlat/corrected-centerlat-clipped.json \
    --clip-bboxes
```

Use `corrected-centerlat-clipped.json` for image-bounded detector evaluation;
preserve `corrected-centerlat-raw.json` as the unmodified reference geometry.

## Visualize COCO predictions

Render prediction masks, bounding boxes, and confidence scores on the source images:

```bash
python tools/dataset/render_coco_prediction_overlays.py \
    /data/mars-runs/sam3/predictions.json \
    /data/mars/corrected \
    /data/mars-runs/sam3/overlays \
    --score-threshold 0.4
```

Use `--limit 8` for a quick sample, `--target IMAGE_NAME` to select specific images, and `--max-detections 100` to cap detections per image.

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
