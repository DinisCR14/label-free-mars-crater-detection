# Project Status

## Current stage

Initial detector training has completed. The spatially isolated training
dataset was prepared from the cached SAM 3 pseudo-labels, and the initial
checkpoint is ready for fixed validation and test evaluation.

The Python environment is split by pipeline stage. `.venv` is reserved for
SAM 3 and other foundation-model work, while `.venv-cutler` is reserved for
CutLER/Detectron2 training. This is required because SAM 3 declares
`iopath>=0.1.10` and Detectron2 0.6 declares `iopath<0.1.10`.

## Selected configuration

- Foundation model: SAM 3
- SAM 3 prediction configuration: confidence threshold `0.4`
- Geometric refinement: enabled
- FBWR threshold: `1500`
- FBWR center-window fraction: `0.10`
- FBWR rim margin: `5`
- FBWR angle count: `12`
- Training split: spatially overlap-aware

Threshold `2000` remains available as a conservative comparison. The complete
FBWR score cache allows additional thresholds to be generated without
recomputing FBWR.

## Verified external artifacts

- SAM 3 source predictions:
  `/home/dcortereal/datasets/sam3-runs/full-corrected-centerlat/sam3-circle-threshold-04.json`
- FBWR score cache:
  `/home/dcortereal/datasets/sam3-runs/scored-corrected-centerlat/fbwr_scored.json`
- Selected pseudo-labels:
  `/home/dcortereal/datasets/sam3-runs/processed-threshold-1500-cached/fbwr_filtered.json`
- Spatial overlap audit:
  `/home/dcortereal/datasets/ground-truth/corrected-centerlat/spatial-overlap-audit.json`
- Prepared detector dataset:
  `/home/dcortereal/datasets/sam3-training-threshold-1500-spatial`

## Verified counts

### Label generation

- Raw SAM 3 predictions: `236,665`
- Geometrically refined predictions: `224,302`
- Threshold-1500 pseudo-labels: `104,003`
- Threshold-2000 pseudo-labels: `61,625`

### Spatial split

- Overlap components: `100`
- Train images: `9,840`
- Validation images: `1,203`
- Test images: `1,221`
- Training pseudo-labels: `96,921`
- Validation reference annotations: `27,244`
- Test reference annotations: `26,709`

## Initial detector baseline

- Checkpoint:
  `/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/training/initial-training/model_final.pth`
- Reported validation bbox metrics:
  - AP: `11.49`
  - AP50: `28.94`
  - AP75: `7.04`
  - AP small: `5.28`
  - AP medium: `26.00`
  - AP large: `29.62`
- Reported validation segmentation metrics:
  - AP: `11.23`
  - AP50: `28.11`
  - AP75: `7.04`
  - AP small: `4.40`
  - AP medium: `28.37`
  - AP large: `28.96`
- Validation predictions:
  `/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/evaluation/initial-validation/inference/coco_instances_results.json`
- Representative prediction/reference overlays:
  `/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/evaluation/initial-validation/overlays/combined-score-0.70`

These are an initial baseline only. The weak small-object performance and
high-confidence false positives make a longer initial-training experiment
necessary before starting self-training.

## Experiment 1: longer initial training

The first improvement experiment keeps the pseudo-labels, architecture,
pretrained weights, spatial split, and evaluation protocol unchanged. It only
increases training from `4,000` to `20,000` iterations using
`cascade_mask_rcnn_R_50_FPN_modified_long.yaml`. This tests whether the
initial detector was undertrained before changing labels or augmentation.

The experiment must use a new output directory:

```text
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/training/initial-training-20k
```

Compare its fixed validation AP, AP50, AP75, and size-stratified metrics with
the baseline before considering self-training. Do not use reference
annotations to alter the training labels.

The 20k run completed on 2026-11-05. Its final checkpoint metrics were:

- Checkpoint:
  `/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/training/initial-training-20k/initial-training/model_final.pth`
- Training duration: approximately `1:23:52`
- Final validation bbox metrics:
  - AP: `11.15`
  - AP50: `26.41`
  - AP75: `8.02`
  - AP small: `4.63`
  - AP medium: `27.20`
  - AP large: `32.35`
- Final validation segmentation metrics:
  - AP: `10.83`
  - AP50: `25.98`
  - AP75: `7.87`
  - AP small: `3.82`
  - AP medium: `28.16`
  - AP large: `32.31`

Compared with the 4k baseline, the final 20k checkpoint has lower overall AP,
AP50, and small-object AP, while AP75 and large-object AP improved slightly.
The intermediate checkpoints at 5k, 10k, and 15k must be evaluated before
concluding that longer training is ineffective.

### Experiment 1 checkpoint sweep

All four intermediate checkpoints were evaluated on the fixed spatial
validation set. The bbox results are:

| Checkpoint | AP | AP50 | AP75 | AP small | AP medium | AP large |
|---|---:|---:|---:|---:|---:|---:|
| 4k baseline | 11.49 | 28.94 | 7.04 | 5.28 | 26.00 | 29.62 |
| 5k | 9.27 | 24.88 | 5.54 | 4.10 | 22.68 | 29.13 |
| 10k | 10.99 | 26.31 | 7.79 | 4.59 | 26.68 | 32.26 |
| 15k | 11.79 | 26.59 | 8.79 | 4.85 | 28.62 | 31.80 |
| 20k | 11.15 | 26.41 | 8.02 | 4.63 | 27.20 | 32.35 |

The 15k checkpoint is the best longer-training checkpoint by overall AP and
AP75, but the 4k baseline remains better on AP50 and small-crater AP. The
longer run therefore improves stricter localization and larger craters
without solving the small-crater or high-confidence false-positive problem.
The evaluation outputs are stored under:

```text
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/evaluation/initial-validation-{5k,10k,15k,20k}/
```

Matched 4k and 15k prediction/reference overlays at score threshold `0.70`
are stored at:

```text
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/evaluation/initial-validation-4k/overlays/combined-score-0.70/
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/evaluation/initial-validation-15k/overlays/combined-score-0.70/
```

At score `>= 0.70`, greedy one-to-one bbox matching gives approximate
precision of `41.9%` at IoU `0.50` and `17.3%` at IoU `0.75` for 4k, compared
with `45.3%` and `20.6%` for 15k. This confirms a modest localization and
precision improvement, but not a solution to the false-positive problem.

## High-confidence prediction triage

The 15k validation predictions were joined to the full SAM 3 pseudo-label
catalogue by `file_name`, not numeric image ID, because the prepared split
uses split-local IDs. At score `>= 0.95`, there are `6,039` predictions:

- `4,714` overlap a reference box at IoU `>= 0.50`;
- `2,685` overlap a SAM 3 pseudo-label at IoU `>= 0.50`;
- `484` have no reference overlap but do overlap a SAM 3 pseudo-label;
- `841` have neither reference nor pseudo-label overlap.

The pseudo-label overlap does not prove that a prediction is correct: it may
represent an omitted reference crater or an inherited pseudo-label error.
However, it shows that the high-confidence error population has both an
inherited-label component and a detector-only component. Representative
triage overlays and confidence-bin metrics are stored externally under:

```text
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/analysis/high-confidence-triage/
```

## Annotation policy

Training uses SAM 3 pseudo-labels only:

```text
train_initial.json
  <- SAM 3 pseudo-labels
```

Reference annotations are used only for offline validation and test
evaluation:

```text
val_truegt.json
test_truegt_clipped.json
```

Reference ground truth must not be used for pseudo-label generation, FBWR
threshold selection, or training annotations.

## Spatial split policy

The corrected images can have overlapping geographic footprints because the
cylindrical-projection correction widens tiles away from the equator. The
dataset preparation code computes corrected footprints, builds connected
overlap components, and keeps each component entirely within one split.

The default preparation strategy is:

```text
--split-strategy spatial
```

The random image-level split remains available only for legacy comparisons:

```text
--split-strategy random
```

## FBWR score-cache policy

The score cache contains scores for all geometrically refined annotations. It
is valid for the exact refined prediction set and geometry used to create it:

```text
center_window_fraction = 0.10
rim_margin              = 5
num_angles              = 12
```

Changing only the FBWR threshold does not require recomputing FBWR. Changing
the geometry or the candidate prediction set requires a new score cache.

## Completed implementation work

- Manifest-based image mapping for model predictions.
- Model-aware label-generation orchestration.
- Geometric refinement and FBWR filtering.
- Vectorized Torch FBWR scoring.
- FBWR score caching and threshold-only filtering.
- FBWR filter-effect and size analysis.
- Spatial overlap audit and overlap-aware dataset splitting.
- Progress reporting for long-running stages.

## Next steps

1. Compare 4k and 15k representative validation overlays, with emphasis on
   false positives, localization, and small-crater recall.
2. Select the checkpoint and training strategy for the next controlled
   experiment; do not start self-training yet.
3. Use `.venv` for label generation and dataset preparation, and
   `.venv-cutler` for detector training. `run_full_pipeline.py` accepts
   `--label-python-executable` and `--training-python-executable` for this
   split.
4. Run the copy-paste-disabled controlled experiment using
   `cascade_mask_rcnn_R_50_FPN_modified_no_copy_paste.yaml`. Compare its 15k
   checkpoint with the existing 15k checkpoint before changing labels or
   starting self-training.
5. Run self-training rounds only if the improved initial detector is viable.
6. Train and evaluate the final standard-loss detector.
7. Compare detector results against the threshold-2000 and other cached
   pseudo-label variants if needed.
