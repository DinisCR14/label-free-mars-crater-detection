# Project Status

## Current stage

Balanced retraining of the cumulative DropLoss, no-DropLoss, and supervised
policies has completed. Fresh balanced validation and test inference has also
completed. A matched Grounded SAM cumulative DropLoss comparison has now
completed, and the next phase is robustness, latitude-stratified analysis, and
benchmark comparison using the selected foundation model.

The Python environment is split by pipeline stage. `.venv` is reserved for
SAM 3 and other foundation-model work, while `.venv-cutler` is reserved for
CutLER/Detectron2 training. This is required because SAM 3 declares
`iopath>=0.1.10` and Detectron2 0.6 declares `iopath<0.1.10`.

## Selected configuration

- Foundation model: Grounded SAM (Grounding DINO plus SAM ViT-H)
- Grounded SAM prediction configuration: box threshold `0.30`, text threshold
  `0.25`, prompt `circle`
- Geometric refinement: enabled
- FBWR threshold: `1500`
- FBWR center-window fraction: `0.10`
- FBWR rim margin: `5`
- FBWR angle count: `12`
- Training split: spatially overlap-aware

The original spatial split is a latitude-shift evaluation: its training
images span approximately -70 to +38 degrees, validation spans +40 to +52
degrees, and test spans +52 to +66 degrees. A separate balanced split has
been prepared for primary evaluation using the
`spatial_latitude_stratified` strategy with a 70/15/15 target. It keeps all
100 overlap groups intact and distributes them across 20-degree latitude
bands.

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
- Balanced 70/15/15 detector dataset:
  `/home/dcortereal/datasets/sam3-training-threshold-1500-balanced-70-15-15`

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

### Balanced split

- Train images: `8,511` (`69.4%`)
- Validation images: `1,782` (`14.5%`)
- Test images: `1,971` (`16.1%`)
- Latitude range: approximately `-70` to `+66` degrees in each split
- Overlap groups: `100`, all kept within one split

### Balanced policy retraining and final evaluation

The cumulative DropLoss, no-DropLoss, and supervised policies were retrained
on the balanced split with the same Cascade Mask R-CNN architecture,
40,000-iteration budget, and standard-loss final consolidation. The
label-free policies started from the same balanced initial pseudo-labels and
used self-training thresholds `0.70`, `0.65`, and `0.60`.

Final checkpoints:

- Cumulative DropLoss:
  `/home/dcortereal/datasets/sam3-training-threshold-1500-balanced-70-15-15/training-cumulative-droploss/final-standard-loss/model_final.pth`
- No-DropLoss:
  `/home/dcortereal/datasets/sam3-training-threshold-1500-balanced-70-15-15/training-no-droploss/final-standard-loss/model_final.pth`
- Supervised:
  `/home/dcortereal/datasets/sam3-training-threshold-1500-balanced-70-15-15/training-supervised/model_final.pth`

Fresh inference was run with `TEST.DETECTIONS_PER_IMAGE=100`. The following
standard COCO bbox metrics use the balanced validation and test references;
the test reference uses clipped boxes. Values are AP@[0.50:0.95], AP50, AP75,
and AR100 respectively:

| Policy | Validation AP / AP50 / AP75 / AR100 | Test AP / AP50 / AP75 / AR100 |
| --- | --- | --- |
| Cumulative DropLoss | 20.99 / 50.83 / 14.99 / 29.35 | 21.58 / 51.96 / 15.62 / 30.46 |
| No-DropLoss | 17.61 / 41.43 / 12.34 / 24.24 | 17.52 / 41.22 / 12.20 / 24.76 |
| Supervised | 36.34 / 66.94 / 33.03 / 42.42 | 37.12 / 67.55 / 34.30 / 43.39 |

Evaluation reports and predictions are external to the repository under:

`/home/dcortereal/datasets/evaluation/balanced-70-15-15-final`

These are the primary balanced-split detector results. Confidence-threshold
calibration and greedy precision/recall/F1 analysis are reported below, with
all thresholds selected using validation only before evaluating the test set.

### Balanced confidence calibration and frozen test operating points

Confidence thresholds were calibrated independently for each policy using
only balanced validation predictions, maximizing score-ranked greedy bbox F1
at IoU `0.50`. Predictions were capped at `100` per image, and thresholds
were searched from `0.00` through `0.99` in increments of `0.01`. The selected
thresholds were then frozen and applied once to the balanced test predictions.

| Policy | Selected threshold | Validation precision / recall / F1 | Test precision / recall / F1 |
| --- | ---: | --- | --- |
| Cumulative DropLoss | 0.76 | 67.33% / 49.04% / 56.75% | 66.23% / 50.32% / 57.19% |
| No-DropLoss | 0.03 | 69.15% / 41.72% / 52.04% | 67.49% / 41.87% / 51.68% |
| Supervised | 0.42 | 81.65% / 64.17% / 71.86% | 80.21% / 65.18% / 71.92% |

The no-DropLoss threshold is low because its score distribution is more
conservative; it is not comparable to the cumulative policy's threshold as a
confidence-calibration value. The frozen test results remain supplementary
operating-point measurements and are separate from COCO AP.

External calibration reports, including the complete validation threshold
curves and source paths, are stored under:

`/home/dcortereal/datasets/evaluation/balanced-70-15-15-final/threshold-calibration`

### Cumulative DropLoss improvement test series

The current balanced cumulative DropLoss result is the locked baseline:
validation AP `20.99`, AP50 `50.83`; test AP `21.58`, AP50 `51.96`.
The balanced test set must not be used to select an improvement. The following
gated series is intended to identify the source of the largest recoverable
gain before starting multi-seed robustness experiments.

#### Diagnostic gate: identify the dominant error

Run without retraining:

1. **Validation stratification** — report AP/AP50 and frozen-threshold
   precision, recall, and F1 by 20-degree latitude band and by crater size.
   The initial operating-point audit shows the largest weakness in the
   northern bands (`+30` to `+70` degrees) and very low large-crater
   precision. The audit is external at:
   `/home/dcortereal/datasets/evaluation/balanced-70-15-15-final/diagnostic-strata/validation_operating_strata.json`.
2. **Pseudo-label source audit** — match initial and final cumulative
   pseudo-labels to the training reference at bbox IoU `0.50`, separately by
   latitude and size. Measure reference coverage, supported precision,
   unsupported detections, and duplicate/overlapping detections.
3. **Round trajectory audit** — evaluate the initial detector and each
   cumulative self-training round on the same validation split. Record AP,
   AP50, size AP, latitude AP, prediction count, and unsupported-prediction
   rate. This distinguishes gains from pseudo-label accumulation from gains
   caused by the final consolidation.

The diagnostic gate determines which branch of the ablation matrix is worth
training. It does not change the baseline or inspect the test annotations.

#### Cheap pseudo-label ablations: no detector retraining

Use the cached SAM 3/FBWR predictions to construct candidate training labels
without recomputing foundation-model inference:

| Candidate | Hypothesis | Proceed if validation evidence shows |
| --- | --- | --- |
| FBWR `1500` baseline | Current balance of coverage and noise | Reference coverage is the limiting factor |
| FBWR `2000` | Stricter filtering reduces accumulated false positives | Unsupported detections dominate and recall loss is acceptable |
| Confidence-aware `1500` | High FBWR scores should use a lower confidence floor, while marginal scores need stricter confidence | Size/latitude slices show low-quality marginal labels |
| Supported-source restriction | Remove labels with no source/shape support | Unsupported cumulative detections are concentrated in a specific slice |

These are label-quality tests, not final performance claims. A candidate is
eligible for retraining only if it improves the validation pseudo-label
coverage/precision trade-off without eliminating the small-crater or
northern-latitude populations.

#### Controlled retraining gate

Retrain at most two candidates, keeping the architecture, split, random seed,
iteration budget, augmentation, and three-round schedule fixed:

1. the candidate with the best pseudo-label coverage/quality trade-off;
2. one alternative targeting the dominant latitude/size failure mode.

For each candidate, compare against the locked cumulative baseline using:

- validation AP and AP50 as the primary selection metrics;
- AP by size and latitude as explanatory metrics;
- frozen validation F1 at IoU `0.50`;
- unsupported-prediction rate and prediction count.

An improvement is considered substantial enough to justify a fresh test
evaluation only if it achieves at least `+2.0` validation AP or `+3.0`
validation AP50, while not losing more than `1.0` AP in any major size or
latitude stratum. If no candidate meets this gate, retain the current
cumulative policy and proceed to robustness/latitude analysis instead of
overfitting the validation split.

#### Locked test decision

Only the best validation-supported candidate receives one fresh test
evaluation. The test comparison must use the same COCO protocol and the
validation-selected confidence threshold. All rejected candidates remain
validation-only diagnostics.

The first pseudo-label diagnostic found that, on the balanced training image
set at bbox IoU `0.50`, the FBWR threshold trade-off is:

| FBWR threshold | Pseudo-labels | Reference-matched precision | Reference coverage |
| ---: | ---: | ---: | ---: |
| 1,200 | 90,626 | 83.2% | 22.5% |
| 1,500 (baseline) | 71,299 | 85.6% | 18.3% |
| 2,000 | 41,926 | 88.4% | 11.1% |

Threshold `1,200` is therefore the first promising candidate: it increases
reference coverage by approximately 23% relative to the baseline while
retaining a substantial matched-label precision. This is a pseudo-label
diagnostic, not detector performance; it must still be validated through
controlled retraining.

The shared balanced `annotations/train_r3.json` was later consumed by the
no-DropLoss run and cannot be used as provenance for cumulative DropLoss
Round 3. Cumulative and no-DropLoss branch-specific detector predictions
remain available, but future pseudo-label audits must use branch-specific
annotation artifacts or regenerate them from the saved inference outputs.

The controlled FBWR-`1200` candidate completed its full three-round
self-training schedule and final standard-loss consolidation. On balanced
validation it achieved AP `23.18` and AP50 `53.02`, compared with the
baseline cumulative policy's AP `20.99` and AP50 `50.83`. However, its
size-stratified standard AP changed as follows:

| Policy | AP small | AP medium | AP large |
| --- | ---: | ---: | ---: |
| Cumulative baseline | 20.58 | 47.42 | 18.55 |
| FBWR-1200 candidate | 22.64 | 44.64 | 15.85 |

The candidate is therefore validation-promising in aggregate but fails the
predefined improvement gate because medium and large AP regress by more than
one point. It remains validation-only and must not be evaluated on the locked
test set. Its external artifacts are under:

`/home/dcortereal/datasets/evaluation/balanced-70-15-15-final/candidate-fbwr1200`

Two further validation candidates were tested:

| Candidate | Stage | AP | AP50 | AP small | AP medium | AP large |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Strict merge, thresholds `0.80/0.75/0.70` | Final | 21.94 | 52.29 | 21.7 | 45.5 | 18.3 |
| FBWR `1350` | Initial only | 17.39 | 42.31 | 15.8 | 46.7 | 20.0 |

Strict merge is slightly better in aggregate than the cumulative baseline,
but it fails the improvement gate because medium-crater AP falls by about
`1.9` points. FBWR `1350` is below the baseline at the initial stage and was
not advanced to full self-training. Neither candidate is eligible for test
evaluation.

The size-aware hybrid candidate (FBWR `1200` for small boxes and `1500` for
medium/large boxes) was also evaluated at the initial-detector gate. Compared
with the baseline initial detector, it achieved AP `15.03` versus `14.35`
and AP50 `39.70` versus `36.88`, while AP small changed from `13.04` to
`14.15`, AP medium from `44.55` to `44.42`, and AP large from `14.71` to
`14.35`. The overall AP gain of `+0.68` is below the `+2.0` retraining gate,
so the hybrid does not proceed to self-training or test evaluation. Its
validation-only artifacts are under:

`/home/dcortereal/datasets/evaluation/balanced-70-15-15-final/candidate-hybrid-small1200`

The stricter cumulative merge was completed using the FBWR-`1500` initial
labels and confidence thresholds `0.80/0.75/0.70` for newly added detector
predictions across the three rounds. It tested whether lower-confidence
prediction accumulation was causing the medium/large regressions seen with
FBWR-`1200`. Its external artifacts are under:

`/home/dcortereal/datasets/evaluation/balanced-70-15-15-final/candidate-strict-merge`

### Round-by-round validation trajectory

The cumulative baseline and FBWR-`1200` candidate were evaluated at the
initial checkpoint, after each self-training round, and after final standard
loss consolidation. All predictions use the balanced validation split and
`TEST.DETECTIONS_PER_IMAGE=100`; the locked test set was not used.

| Policy / stage | AP | AP50 | AP75 | AP small | AP medium | AP large | AR100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cumulative / initial | 14.29 | 36.93 | 8.74 | 13.09 | 44.44 | 14.66 | 20.73 |
| Cumulative / R1 | 18.07 | 46.35 | 11.69 | 17.50 | 45.87 | 17.06 | 25.82 |
| Cumulative / R2 | 20.47 | 50.71 | 13.87 | 20.13 | 45.01 | 16.79 | 28.47 |
| Cumulative / R3 | 21.90 | 52.09 | 15.73 | 21.77 | 44.98 | 16.86 | 29.87 |
| Cumulative / final | 20.99 | 50.83 | 14.99 | 20.58 | 47.42 | 18.55 | 29.35 |
| FBWR-1200 / initial | 18.33 | 43.38 | 13.17 | 17.06 | 46.99 | 20.30 | 23.74 |
| FBWR-1200 / R1 | 20.44 | 49.63 | 14.26 | 19.84 | 46.40 | 19.54 | 27.54 |
| FBWR-1200 / R2 | 22.57 | 52.66 | 16.70 | 21.99 | 45.64 | 17.99 | 29.55 |
| FBWR-1200 / R3 | 22.66 | 53.23 | 16.68 | 22.26 | 44.52 | 16.90 | 30.08 |
| FBWR-1200 / final | 23.18 | 53.02 | 17.62 | 22.64 | 44.64 | 15.85 | 31.07 |

The cumulative trajectory shows that the largest aggregate gains arrive by
Round 2 and Round 3, while final standard-loss consolidation lowers AP by
`0.91` points from the R3 checkpoint but improves medium and large AP. The
FBWR-`1200` trajectory continues improving through final consolidation, but
its final medium and large AP remain below the cumulative baseline. This
confirms that the candidate's aggregate gain is driven primarily by small
craters and does not satisfy the balanced size gate.

The consolidated report is stored externally at:

`/home/dcortereal/datasets/evaluation/balanced-70-15-15-final/round-evals/metrics/coco_bbox_metrics.json`

The cumulative Round 4 continuation was then run from the saved cumulative
Round 3 checkpoint using confidence threshold `0.55` for newly added
predictions. It was evaluated on balanced validation only, without final
standard-loss consolidation and without accessing the test set:

| Policy / stage | AP | AP50 | AP75 | AP small | AP medium | AP large | AR100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cumulative / R4 | 23.04 | 53.18 | 16.76 | 22.64 | 44.01 | 14.89 | 31.17 |

Relative to the cumulative final baseline (`20.99 / 50.83`), Round 4 gains
`+2.04` AP and `+2.35` AP50, and therefore passes the aggregate improvement
threshold. It fails the balanced size gate, however: medium AP decreases by
`3.41` points and large AP decreases by `3.66` points, while the small-crater
gain is `+2.06`. Round 4 is consequently a useful saturation diagnostic but
is not eligible for locked-test evaluation or replacement of the cumulative
final baseline. Its external artifacts are under:

`/home/dcortereal/datasets/evaluation/balanced-70-15-15-final/round4-cumulative`

## Matched Grounded SAM comparison

Grounded SAM was evaluated with the same balanced 70/15/15 split, FBWR-1500
pseudo-label threshold, Cascade Mask R-CNN architecture, cumulative
DropLoss schedule, three self-training rounds, final standard-loss
consolidation, and 100 detections per image as the selected SAM 3 baseline.
Grounded SAM used Grounding DINO (`IDEA-Research/grounding-dino-base`) with
the official SAM ViT-H checkpoint. Its label-generation run produced 5,360
FBWR-1500 pseudo-labels, compared with 104,003 for SAM 3; this difference is
an important limitation when interpreting the matched detector comparison.

| Foundation model | Validation AP | Validation AP50 | Validation AP75 | AP small | AP medium | AP large | AR100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAM 3 | 20.99 | 50.83 | 14.99 | 20.58 | 47.42 | 18.55 | 29.35 |
| Grounded SAM | 24.46 | 52.60 | 19.35 | 22.61 | 49.30 | 26.24 | 31.49 |

Grounded SAM clears the predeclared validation gate with a `+3.47` AP gain
over SAM 3 and no regression in standard COCO size AP. It therefore received
the one authorized locked-test evaluation:

| Foundation model | Test AP | Test AP50 | Test AP75 | AP small | AP medium | AP large | AR100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAM 3 | 21.58 | 51.96 | 15.62 | 21.22 | 47.31 | 16.86 | 30.46 |
| Grounded SAM | 25.12 | 53.89 | 20.11 | 23.46 | 48.53 | 24.72 | 32.56 |

Grounded SAM is now the selected foundation model for subsequent analyses.
The SAM 3 results remain the matched comparator, and no additional
foundation-model or policy selection should be performed using the locked
test set.

### Initial pseudo-label quality comparison

The initial pseudo-label sets were compared on the identical 8,511-image
balanced training split. Matches use greedy one-to-one bbox IoU at `0.50`
against the training reference annotations; the reference is used only for
this diagnostic, not for pseudo-label generation.

| Foundation model | Initial labels | Images with labels | Matched labels | Supported precision | Reference coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| SAM 3 | 71,299 | 7,307 | 60,960 | 85.50% | 18.23% |
| Grounded SAM | 3,704 | 3,263 | 3,659 | **98.79%** | 1.09% |

Grounded SAM produced only 5.2% as many training pseudo-labels as SAM 3, but
its unsupported-label count was just `45`, compared with `10,339` for SAM 3.
The two sets had `3,515` matched labels in common: this is 94.90% of the
Grounded SAM set but only 4.93% of the SAM 3 set. Grounded SAM therefore acts
like a highly precise subset of the broader SAM 3 proposal set rather than a
different population of craters.

The quality difference is also size- and latitude-dependent. SAM 3's
supported precision was 71.2% for small labels, 88.9% for medium labels, and
99.0% for large labels. Grounded SAM produced no labels in the project's
small bbox-area band, while its medium and large label precision was 100.0%
and 98.2%, respectively. SAM 3's supported precision declined from 91.1% in
the southernmost band (`-70` to `-50` degrees) to 61.3% in the northernmost
band (`+50` to `+70` degrees); Grounded SAM remained between 97.3% and 99.8%
across the latitude bands.

This explains why label quantity alone did not predict detector quality:
SAM 3 supplied much broader coverage but also substantially more false
positives, especially for small and northern-latitude proposals. Grounded
SAM supplied sparse, high-confidence supervision, after which detector
self-training recovered additional proposals. The coverage difference remains
a limitation and should be included in the robustness and ablation analyses.

### Precision-coverage ablation

Two validation-only initial-label ablations were trained with the same
balanced split, architecture, pretrained weights, cumulative DropLoss
schedule, self-training thresholds, and final consolidation. The locked test
set was not used.

| Initial-label policy | Training labels | Validation AP | AP50 | AP75 | AP small | AP medium | AP large | AR100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SAM 3 FBWR-1500 baseline | 71,299 | 20.99 | 50.83 | 14.99 | 20.58 | 47.42 | 18.55 | 29.35 |
| SAM 3 FBWR-2000 | 41,926 | 19.62 | 49.75 | 12.65 | 19.58 | 39.10 | 6.55 | 27.53 |
| Grounded SAM FBWR-1500 | 3,704 | 24.46 | 52.60 | 19.35 | 22.61 | 49.30 | 26.24 | 31.49 |
| Grounded SAM + SAM 3 FBWR-2000 | 43,294 | 21.47 | 51.43 | 15.06 | 20.83 | 40.72 | 14.99 | 28.92 |

The strict SAM 3 policy is worse than the FBWR-1500 baseline despite having
fewer and cleaner labels. Its large-crater AP collapses, showing that
precision cannot be optimized independently of coverage and size composition.
The hybrid recovers some small-crater performance relative to strict SAM 3,
but remains well below Grounded SAM and loses 8.58 AP on medium craters and
11.25 AP on large craters relative to it.

Neither candidate satisfies the improvement gate. The hybrid's aggregate
gain over the SAM 3 baseline is only `+0.48` AP and its medium and large AP
regress by more than `1.0`; relative to the selected Grounded SAM policy it
is `-3.00` AP. Both candidates are therefore rejected and receive no test
evaluation. The selected Grounded SAM cumulative policy remains unchanged.

External ablation artifacts:

- Strict SAM 3 dataset:
  `/home/dcortereal/datasets/sam3-training-threshold-2000-balanced-70-15-15`
- Hybrid dataset:
  `/home/dcortereal/datasets/hybrid-grounded1500-sam32000-training-balanced-70-15-15`
- Validation metrics:
  `/home/dcortereal/datasets/evaluation/balanced-70-15-15-final/pseudo-label-ablation-metrics-val/coco_bbox_metrics.json`

External Grounded SAM artifacts:

- Labels:
  `/home/dcortereal/datasets/grounded-sam-runs/balanced-comparison-threshold-1500`
- Balanced detector dataset:
  `/home/dcortereal/datasets/grounded-sam-training-threshold-1500-balanced-70-15-15`
- Validation metrics:
  `/home/dcortereal/datasets/evaluation/balanced-70-15-15-final/grounded-sam-cumulative-val/metrics-val/coco_bbox_metrics.json`
- Test metrics:
  `/home/dcortereal/datasets/evaluation/balanced-70-15-15-final/grounded-sam-cumulative-test/metrics-test/coco_bbox_metrics.json`

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

### Round 3 branch comparison

The two complete three-round self-training branches were evaluated on the
fixed validation set:

| Branch | AP | AP50 | AP75 | AP small | AP medium | AP large |
|---|---:|---:|---:|---:|---:|---:|
| Cumulative Drop Loss | 17.20 | 38.56 | 13.36 | 10.41 | 35.72 | 29.18 |
| No-DropLoss | 14.65 | 31.12 | 11.57 | 7.12 | 33.05 | 29.33 |

The cumulative Drop Loss branch improves AP by `2.55` points, AP50 by `7.43`
points, AP75 by `1.80` points, and small-crater AP by `3.29` points.
However, its confidence behavior diverges further:

| Confidence threshold | Configuration | Predictions | Reference IoU >= 0.50 | Pseudo-label IoU >= 0.50 | Neither source |
|---|---|---:|---:|---:|---:|
| >= 0.70 | Cumulative Drop Loss | 43,493 | 13,139 (30.2%) | 3,688 | 29,651 |
| >= 0.70 | No-DropLoss | 8,268 | 6,179 (74.7%) | 3,403 | 1,664 |
| >= 0.90 | Cumulative Drop Loss | 29,881 | 12,191 (40.8%) | 3,625 | 17,043 |
| >= 0.90 | No-DropLoss | 6,259 | 5,248 (83.8%) | 3,256 | 673 |
| >= 0.95 | Cumulative Drop Loss | 19,709 | 10,889 (55.2%) | 3,538 | 8,250 |
| >= 0.95 | No-DropLoss | 5,446 | 4,756 (87.3%) | 3,141 | 398 |

The cumulative branch therefore has the strongest detector metrics but a very
large unsupported high-confidence population. These unmatched counts remain
provisional because bbox matching can miss valid ellipse matches and the
reference catalogue may be incomplete. Final policy selection must therefore
include qualitative/catalogue review and final standard-loss consolidation,
not AP alone.

A focused visual review set of 48 high-confidence predictions produced by the
Drop Loss model but not matched by the no-DropLoss model is stored at:

```text
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/round1-pilot/qualitative/drop-loss-extra/
```

The review sheets draw Drop Loss-only predictions in red, no-DropLoss
predictions in green, and reference boxes in blue. These are selection aids,
not correctness labels; catalogue omissions and bbox/ellipse localization
differences still require separate review.

### Final standard-loss consolidation

Both Round 3 branches were consolidated with the matched 40,000-iteration
standard-loss configuration (`USE_DROPLOSS: False`) and evaluated without
changing the validation or test protocols.

Final validation bbox results:

| Branch | AP | AP50 | AP75 | AP small | AP medium | AP large |
|---|---:|---:|---:|---:|---:|---:|
| Cumulative Drop Loss | 17.20 | 38.40 | 13.10 | 10.20 | 35.70 | 29.60 |
| No-DropLoss | 14.50 | 30.50 | 11.70 | 6.80 | 33.60 | 30.30 |

Final held-out test bbox results:

| Branch | AP | AP50 | AP75 | AP small | AP medium | AP large |
|---|---:|---:|---:|---:|---:|---:|
| Cumulative Drop Loss | 12.85 | 26.66 | 11.11 | 4.83 | 31.56 | 32.11 |
| No-DropLoss | 12.23 | 23.98 | 11.01 | 4.08 | 31.06 | 33.55 |

The exact Detectron2 outputs are stored at:

```text
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/final-pilot-cumulative-droploss/evaluation-val/
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/final-pilot-no-droploss/evaluation-val/
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/final-pilot-cumulative-droploss/evaluation-test/
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/final-pilot-no-droploss/evaluation-test/
```

The final source-match audit at bbox IoU `>= 0.50` is:

| Split | Branch | Confidence | Predictions | Reference overlap | Pseudo-label overlap | Neither source |
|---|---|---:|---:|---:|---:|---:|
| Validation | Cumulative Drop Loss | >= 0.95 | 9,895 | 7,573 (76.5%) | 3,377 | 1,903 |
| Validation | No-DropLoss | >= 0.95 | 5,109 | 4,520 (88.5%) | 3,073 | 308 |
| Test | Cumulative Drop Loss | >= 0.95 | 8,057 | 5,532 (68.7%) | 2,754 | 2,171 |
| Test | No-DropLoss | >= 0.95 | 4,297 | 3,881 (90.3%) | 2,500 | 262 |

Standard-loss consolidation preserves the cumulative branch's aggregate AP
advantage, especially on validation and small craters, and gives it a modest
test AP advantage. The no-DropLoss branch is substantially more conservative
and better aligned with both reference and pseudo-label sources at high
confidence, while remaining competitive on test AP and stronger on test large
craters. The final policy should therefore be selected according to the
intended operating point: cumulative Drop Loss for maximum catalogue recall
after catalogue review, or no-DropLoss for a more reliable high-confidence
detector. Neither choice should be justified by AP alone.

### Final training-set label counts

The final standard-loss models use the branch-specific Round 3 annotation
files on the same `9,840` training images:

| Training policy | Pseudo-label annotations | Canonical reference annotations | Pseudo/reference |
|---|---:|---:|---:|
| Cumulative Drop Loss | 279,424 | 429,510 | 65.1% |
| No-DropLoss | 166,604 | 429,510 | 38.8% |

The reference count is obtained by joining the canonical corrected-centerlat
catalogue to the fixed training split. It is evaluation-only and was not used
to construct either final training annotation file. The branch-specific
annotation files are:

```text
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/round3-pilot-cumulative-droploss/annotations/train_r3.json
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/round3-pilot-no-droploss/annotations/train_r3.json
```

### Final precision-recall-F1 diagnostics

The score-ranked greedy bbox diagnostics use IoU thresholds from `0.1` to
`0.9`, cap detections at `100` per image, and report precision, recall, and F1
for all detections and for the project's small, medium, and large bbox-area
ranges. The generated external reports and figures are:

```text
/home/dcortereal/datasets/evaluation/final-policy/val/greedy_bbox_metrics.json
/home/dcortereal/datasets/evaluation/final-policy/val/greedy_bbox_metrics_all.png
/home/dcortereal/datasets/evaluation/final-policy/val/greedy_bbox_metrics_by_size.png
/home/dcortereal/datasets/evaluation/final-policy/test/greedy_bbox_metrics.json
/home/dcortereal/datasets/evaluation/final-policy/test/greedy_bbox_metrics_all.png
/home/dcortereal/datasets/evaluation/final-policy/test/greedy_bbox_metrics_by_size.png
```

For the primary cumulative policy, the inference confidence threshold was
selected on validation only by maximizing greedy bbox F1 at IoU `0.50`, with
100 detections per image. The selected threshold is `0.68`:

| Split | Threshold | Precision | Recall | F1 | Predictions |
|---|---:|---:|---:|---:|---:|
| Validation | 0.68 | 0.5255 | 0.3863 | 0.4453 | 20,027 |
| Test | 0.68 | 0.3876 | 0.2803 | 0.3253 | 19,317 |

The test row was computed only after freezing the threshold selected from
validation. It is a supplementary operating-point result and is distinct
from the COCO AP metrics above. The calibration and test operating-point
reports are stored at:

```text
/home/dcortereal/datasets/evaluation/final-policy/cumulative-threshold-calibration.json
/home/dcortereal/datasets/evaluation/final-policy/cumulative-test-at-threshold-0.68.json
```

### Matched supervised comparator

A matched supervised comparator was trained with the canonical reference
annotations on the same `9,840` training images, using the same Cascade Mask
R-CNN architecture, 40,000-iteration budget, preprocessing, augmentation,
and final standard-loss configuration. Only the training annotations differ
from the label-free policies.

The supervised comparator achieved:

| Split | AP | AP50 | AP75 | AP small | AP medium | AP large |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 29.30 | 51.31 | 27.91 | 17.46 | 50.87 | 45.26 |
| Test | 24.09 | 41.84 | 23.30 | 12.87 | 44.22 | 41.05 |

Using the same validation-only F1 calibration protocol at bbox IoU `0.50`,
the supervised operating threshold is `0.45`:

| Split | Threshold | Precision | Recall | F1 | Predictions |
|---|---:|---:|---:|---:|---:|
| Validation | 0.45 | 0.6076 | 0.5391 | 0.5713 | 24,172 |
| Test | 0.45 | 0.5671 | 0.4266 | 0.4869 | 20,093 |

The supervised comparator is therefore a substantially stronger reference
point than either label-free branch. This comparison does not invalidate the
label-free result: the comparator uses the full reference catalogue during
training, whereas the label-free policies use only foundation-model
pseudo-labels. The evaluation outputs and threshold reports are stored at:

```text
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/supervised-comparator/training/
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/supervised-comparator/evaluation-val/
/home/dcortereal/datasets/sam3-training-threshold-1500-spatial/supervised-comparator/evaluation-test/
/home/dcortereal/datasets/evaluation/final-policy/supervised-threshold-calibration.json
/home/dcortereal/datasets/evaluation/final-policy/supervised-test-at-threshold-0.45.json
```

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

1. Review the unmatched cumulative-branch predictions qualitatively and
   against additional catalogue sources before selecting the operating policy.
2. Freeze the selected policy, manifests, splits, annotation provenance,
   configurations, environment versions, and evaluation protocol.
3. Establish the matched supervised comparator before starting multi-seed
   robustness and latitude-stratified studies.

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
