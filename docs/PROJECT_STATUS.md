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

1. Run Experiment 1 with the longer initial-training configuration.
2. Evaluate the Experiment 1 checkpoint on the fixed spatial validation set
   and render representative overlays.
3. Use `.venv` for label generation and dataset preparation, and
   `.venv-cutler` for detector training. `run_full_pipeline.py` accepts
   `--label-python-executable` and `--training-python-executable` for this
   split.
4. Run self-training rounds only if the improved initial detector is viable.
5. Train and evaluate the final standard-loss detector.
6. Compare detector results against the threshold-2000 and other cached
   pseudo-label variants if needed.
