# Project Status

## Current stage

The project is ready for initial detector training. The spatially isolated
training dataset has been prepared from the cached SAM 3 pseudo-labels.

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

1. Run an initial-only training dry-run with `--initial-only`.
2. Run initial detector training using `train_initial.json`.
3. Evaluate the initial checkpoint on the fixed spatial validation and test
   sets.
4. Run self-training rounds if the initial detector is viable.
5. Train and evaluate the final standard-loss detector.
6. Compare detector results against the threshold-2000 and other cached
   pseudo-label variants if needed.
