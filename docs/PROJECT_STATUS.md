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

### Drop Loss ablation

The initial detector uses FBWR Drop Loss, but its detector-side FBWR crater
threshold is `0.03`, which is not equivalent to the pseudo-label filtering
threshold of `1500`. The thresholds use different intensity representations:
the pseudo-label filter operates on uint8-derived scores, while the detector
loss operates on normalized grayscale inputs.

To isolate whether Drop Loss contributes to the high-confidence predictions,
the next controlled run keeps the initial 4k schedule, copy-paste augmentation,
pseudo-labels, architecture, and weights unchanged. It changes only:

```yaml
MODEL:
  ROI_HEADS:
    USE_DROPLOSS: False
```

Configuration:

```text
third_party/CutLER/cutler/model_zoo/configs/CutLER-CraterDataset/cascade_mask_rcnn_R_50_FPN_modified_no_droploss.yaml
```

The output must use a separate directory:

```text
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/training/initial-training-no-droploss
```

The 4k Drop Loss-disabled run completed on 2026-11-07. Its validation results
compared with the original 4k baseline are:

| Configuration | AP | AP50 | AP75 | AP small | AP medium | AP large |
|---|---:|---:|---:|---:|---:|---:|
| 4k with Drop Loss | 11.49 | 28.94 | 7.04 | 5.28 | 26.00 | 29.62 |
| 4k without Drop Loss | 10.74 | 26.13 | 7.28 | 4.61 | 25.50 | 28.42 |

Segmentation metrics for the Drop Loss-disabled run were AP `10.25`, AP50
`25.63`, AP75 `7.01`, AP small `3.62`, AP medium `26.52`, and AP large
`28.94`.

Disabling Drop Loss slightly improved strict localization (AP75), but reduced
overall AP, AP50, and performance across all size categories. This does not
support removing Drop Loss as an overall improvement for the initial detector.
The next comparison should therefore focus on whether the disabled-loss model
reduces high-confidence detector-only predictions, rather than treating its
lower aggregate AP as evidence of better behavior.

The fixed validation inference for this ablation is stored at:

```text
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/evaluation/initial-validation-no-droploss/
```

The high-confidence comparison, joined by canonical `file_name`, is:

| Confidence threshold | Configuration | Predictions | Reference IoU >= 0.50 | Pseudo-label IoU >= 0.50 | Neither source |
|---|---|---:|---:|---:|---:|
| >= 0.70 | With Drop Loss | 21,790 | 9,137 (41.9%) | 3,673 | 11,881 |
| >= 0.70 | Without Drop Loss | 7,492 | 5,354 (71.5%) | 3,424 | 1,572 |
| >= 0.90 | With Drop Loss | 14,773 | 8,177 (55.4%) | 3,539 | 5,940 |
| >= 0.90 | Without Drop Loss | 4,399 | 3,770 (85.7%) | 2,987 | 293 |
| >= 0.95 | With Drop Loss | 10,430 | 7,118 (68.2%) | 3,405 | 2,769 |
| >= 0.95 | Without Drop Loss | 3,210 | 2,917 (90.9%) | 2,586 | 79 |

Disabling Drop Loss therefore strongly supports the hypothesis that the
initial Drop Loss configuration contributes to the high-confidence
overprediction behavior. It reduces the number of predictions at confidence
`>= 0.95` by approximately `69%`, while reducing predictions unmatched to both
the reference and pseudo-label sources from `2,769` to `79`. This does not
prove that every removed prediction was a false positive, because the
reference catalogue can be incomplete, but the change is substantially larger
than the small aggregate AP difference alone suggests.

### Round 1 self-training pilot

The next experiment starts from the completed no-DropLoss initial checkpoint.
It compares one self-training round with Drop Loss enabled against a matched
round with Drop Loss disabled. The initial validation inference has already
been completed; its predictions are merged with the original training
pseudo-labels at confidence threshold `0.70`.

The no-DropLoss self-training configuration is:

```text
third_party/CutLER/cutler/model_zoo/configs/CutLER-CraterDataset/cascade_mask_rcnn_R_50_FPN_self_train_modified_no_droploss.yaml
```

Do not launch all three self-training rounds until this Round 1 comparison has
been evaluated.

Both Round 1 checkpoints were evaluated on the fixed validation set. The
results are:

| Round 1 self-training | AP | AP50 | AP75 | AP small | AP medium | AP large |
|---|---:|---:|---:|---:|---:|---:|
| Drop Loss enabled | 12.87 | 29.82 | 9.07 | 6.29 | 28.85 | 29.69 |
| Drop Loss disabled | 12.52 | 28.85 | 8.88 | 5.73 | 28.44 | 29.12 |

At the same time, the confidence/matching audit shows:

| Confidence threshold | Configuration | Predictions | Reference IoU >= 0.50 | Pseudo-label IoU >= 0.50 | Neither source |
|---|---|---:|---:|---:|---:|
| >= 0.70 | Drop Loss enabled | 16,863 | 8,479 (50.3%) | 3,639 | 7,702 |
| >= 0.70 | Drop Loss disabled | 7,258 | 5,462 (75.3%) | 3,387 | 1,326 |
| >= 0.90 | Drop Loss enabled | 10,663 | 6,927 (65.0%) | 3,500 | 3,179 |
| >= 0.90 | Drop Loss disabled | 5,419 | 4,531 (83.6%) | 3,187 | 524 |
| >= 0.95 | Drop Loss enabled | 8,233 | 6,076 (73.8%) | 3,376 | 1,691 |
| >= 0.95 | Drop Loss disabled | 4,628 | 4,021 (86.9%) | 3,038 | 299 |

Round 1 Drop Loss improves aggregate AP by `0.35` points, but it produces
approximately `2.7x` as many predictions at confidence `>= 0.95` and
approximately `5.7x` as many predictions unmatched to both annotation
sources. The no-DropLoss variant therefore remains the safer candidate for
the next round unless recall of catalogue-missing craters is prioritized over
confidence reliability. The evaluation outputs are stored under:

```text
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/round1-pilot/evaluation-with-droploss/
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/round1-pilot/evaluation-without-droploss/
```

### Round 2 self-training comparison

Both Round 2 models started from the same Round 1 no-DropLoss checkpoint and
used the same Round 2 merged annotations, with only Drop Loss differing.
Fixed validation results are:

| Round 2 self-training | AP | AP50 | AP75 | AP small | AP medium | AP large |
|---|---:|---:|---:|---:|---:|---:|
| Drop Loss enabled | 13.62 | 30.76 | 10.24 | 6.84 | 30.56 | 29.36 |
| Drop Loss disabled | 13.65 | 29.93 | 10.53 | 6.60 | 30.81 | 29.17 |

At the same time, the confidence/matching audit shows:

| Confidence threshold | Configuration | Predictions | Reference IoU >= 0.50 | Pseudo-label IoU >= 0.50 | Neither source |
|---|---|---:|---:|---:|---:|
| >= 0.70 | Drop Loss enabled | 16,369 | 8,433 (51.5%) | 3,627 | 7,289 |
| >= 0.70 | Drop Loss disabled | 7,213 | 5,590 (77.5%) | 3,374 | 1,194 |
| >= 0.90 | Drop Loss enabled | 10,380 | 6,904 (66.5%) | 3,487 | 2,953 |
| >= 0.90 | Drop Loss disabled | 5,473 | 4,663 (85.2%) | 3,168 | 476 |
| >= 0.95 | Drop Loss enabled | 8,091 | 6,053 (74.8%) | 3,380 | 1,594 |
| >= 0.95 | Drop Loss disabled | 4,693 | 4,157 (88.6%) | 3,028 | 267 |

Round 2 no-DropLoss is marginally better by overall AP (`13.65` vs `13.62`)
and AP75, while Drop Loss remains better on AP50 and small-crater AP. The
no-DropLoss model continues to produce approximately half as many
high-confidence predictions and far fewer predictions unmatched to both
annotation sources. The Round 2 outputs are stored under:

```text
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/round2-pilot/evaluation-with-droploss/
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/round2-pilot/evaluation-without-droploss/
```

### Cumulative Drop Loss branch

The missing cumulative branch was then trained using its own Round 2
annotations:

```text
Initial no-DropLoss
  -> Round 1 with DropLoss
  -> Round 2 with DropLoss
```

Its fixed validation results were:

| Branch | AP | AP50 | AP75 | AP small | AP medium | AP large |
|---|---:|---:|---:|---:|---:|---:|
| Cumulative Drop Loss | 15.42 | 35.01 | 11.47 | 8.63 | 33.00 | 29.72 |
| Round 2 no-DropLoss branch | 13.65 | 29.93 | 10.53 | 6.60 | 30.81 | 29.17 |

The cumulative Drop Loss branch has substantially better aggregate detection
metrics, but its confidence behavior is much less reliable:

| Confidence threshold | Predictions | Reference IoU >= 0.50 | Pseudo-label IoU >= 0.50 | Neither source |
|---|---:|---:|---:|---:|
| >= 0.70 | 31,375 | 11,506 (36.7%) | 3,685 | 19,158 |
| >= 0.90 | 20,466 | 10,185 (49.8%) | 3,610 | 9,636 |
| >= 0.95 | 14,451 | 8,956 (62.0%) | 3,510 | 4,941 |

This is the clearest evidence so far of a recall/precision-calibration
trade-off. Cumulative Drop Loss improves AP, AP50, AP75, and small-crater AP,
but produces a very large population of high-confidence predictions that
match neither the reference nor filtered pseudo-label catalogue by bbox IoU.
These unmatched predictions require qualitative and catalogue review before
selecting the final self-training policy.

A focused visual review set of 48 high-confidence predictions produced by the
Drop Loss model but not matched by the no-DropLoss model is stored at:

```text
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/round1-pilot/qualitative/drop-loss-extra/
```

The review sheets draw Drop Loss-only predictions in red, no-DropLoss
predictions in green, and reference boxes in blue. These are selection aids,
not correctness labels; catalogue omissions and bbox/ellipse localization
differences still require separate review.

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

## Copy-paste ablation result

The copy-paste-disabled run completed with the same 20k schedule. The
like-for-like 15k checkpoint evaluation is:

| Configuration | AP | AP50 | AP75 | AP small | AP medium | AP large |
|---|---:|---:|---:|---:|---:|---:|
| 15k with copy-paste | 11.79 | 26.59 | 8.79 | 4.85 | 28.62 | 31.80 |
| 15k without copy-paste | 11.73 | 25.14 | 9.32 | 4.79 | 28.05 | 31.03 |

Segmentation metrics for the no-copy-paste 15k checkpoint are AP `11.79`,
AP50 `25.32`, AP75 `9.42`, AP small `4.15`, AP medium `29.42`, and AP large
`31.76`. The no-copy-paste configuration slightly improves strict
localization (AP75) but reduces AP50 and does not improve small-crater AP.
The final 20k no-copy-paste checkpoint is weaker still: bbox AP `9.44`,
AP50 `22.70`, AP75 `6.78`, and AP small `3.77`.

The ablation therefore does not support copy-paste as the sole cause of the
high-confidence false positives. It changes the precision/localization
trade-off, but does not produce a clear overall improvement.

## Forensic catalogue and detector audit

The 15k high-confidence audit was joined by canonical `file_name`. On the
1,203 validation images:

- Reference annotations: `27,244`
- Filtered SAM 3 pseudo-labels: `3,824`
- Greedy reference/pseudo-label matches at bbox IoU `>= 0.50`: `2,656`
- Validation images with no filtered SAM 3 pseudo-label: `321`

This large coverage difference is expected for label-free training: pseudo
labels are proposals, not the evaluation authority, and the detector is
expected to discover craters absent from them. Under the canonical annotation
policy, a prediction that does not match the reference catalogue remains a
candidate false positive unless the reference geometry or mapping audit shows
otherwise. At score `>= 0.95`, the 15k detector has `1,325`
reference-unmatched predictions; `484` overlap a filtered pseudo-label and
`841` overlap neither source. Pseudo-label overlap does not make a prediction
correct.

Among the `841` detector-only cases, the bbox-area diagnostic classified
`289` as small, `524` as medium, `1` as large, and `27` as border-touching
cases (the border subset overlaps the size counts). These are diagnostic
categories, not ground-truth error labels. The raw and clipped canonical
reference files contain the same `483,463` annotations; clipping changes
image-bounded COCO boxes while preserving the reference ellipse geometry.
The next audit must compare the `1,325` unmatched predictions against that
ellipse geometry and the filename mapping before classifying them as
detector errors. Detailed outputs are stored at:

```text
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/analysis/forensic/
```

The first raw-ellipse audit found `612` of the `1,325` bbox-unmatched
predictions to be plausible reference-geometry candidates under a provisional
center-and-scale test. This is not a correctness label; it indicates that
bbox IoU is too coarse for final triage and that localization/ellipse
clipping effects may explain a substantial portion of the apparent
false-positive count. The detailed provisional records are in
`reference_geometry_audit.json` under the forensic directory.

Representative raw-ellipse overlays for manual review are stored in:

```text
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/analysis/forensic/geometry_candidates/
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/analysis/forensic/geometry_non_candidates/
```

The `713` provisional non-candidates are available as a ranked review queue
with score, tile, bbox, size, and geometry diagnostics, plus 20 contact
sheets:

```text
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/analysis/forensic/review/non_candidates_review.csv
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/analysis/forensic/review/non_candidates_sheet_*.jpg
```

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

1. Finish the Round 3 comparison for the cumulative Drop Loss and no-DropLoss
   branches.
2. Train and evaluate the final standard-loss model for the selected branches.
3. Freeze the selected policy, manifests, splits, annotation provenance,
   configurations, and evaluation protocol before starting robustness studies.

## Long-term J-STARS research roadmap

The long-term goal is to extend the IGARSS 2026 work into a J-STARS
submission. The planned extensions are:

1. **Robustness and uncertainty:** run the selected pipeline across multiple
   independent training seeds and report mean, standard deviation, and
   confidence intervals. Use image- or tile-level bootstrap resampling for
   metric intervals because crater annotations within one image are correlated.
2. **Supervised comparison:** train the same detector architecture with the
   canonical reference annotations, keeping the image split, training budget,
   preprocessing, and evaluation protocol fixed. Compare this reference-
   supervised model with the label-free model and the foundation-model
   baseline.
3. **External benchmarks:** investigate Mars Bench and Marsdo16k, documenting
   imagery, resolution, latitude coverage, annotation format, crater-size
   range, split policy, and metric definitions before making comparisons.
   Benchmark labels must remain separate from the canonical project ground
   truth.
4. **Latitude-stratified analysis:** report coverage, crater-size
   distribution, AP/AP50/AP75, recall, precision at fixed confidence
   thresholds, and unsupported-prediction rates across balanced latitude
   bands. Interpret geographic trends together with image quality, crater
   density, and catalogue completeness.

The planned order is: finalize the training policy, freeze the reproducibility
package, establish the supervised comparator, run multi-seed robustness
experiments, perform latitude analysis, and then add external benchmark
comparisons. All comparisons must preserve the manifest-based mapping,
spatial split policy, and strict separation between training pseudo-labels and
evaluation reference annotations.
