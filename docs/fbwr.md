# FBWR design and optimization

Flexible Black and White Rims (FBWR) is used in two different stages of the
pipeline. The stages intentionally use different intensity units and
thresholds.

## Pseudo-label filtering

`crater_detection/filter_coco_fbwr.py` reads the source images as 8-bit
grayscale OpenCV arrays. It uses `FBWRConfig.threshold_uint8`, whose default is
`2000.0`, or a threshold supplied through `--threshold`.

The initial pseudo-label parameters are configurable with
`--center-window-fraction` (default `0.10`), `--rim-margin` (default `5`
pixels), and `--num-angles` (default `12`). These values are recorded in the
audit report together with the threshold.

The score is computed from raw image intensities and is therefore on the raw
uint8-derived scale. A prediction passes only when:

```text
score >= threshold
```

The score is bbox-based: it samples opposite radial points around the bbox
center and combines their absolute intensity contrast with the local gradient
magnitude. Samples for which either endpoint is outside the image are ignored.
This means boundary-touching boxes can have fewer contributing samples than
interior boxes.

The retained COCO annotations include their computed score in the
`fbwr_score` field. The filter also writes a `<output-stem>_report.json` file
containing the threshold, comparison operator, input/retained/discarded
counts, score summaries, and valid-sample-fraction summaries. Retained
annotations also include `fbwr_valid_sample_fraction`. A different report path
can be supplied with `--report-json`.

The score remains an average over valid samples, so its intensity scale is
comparable for interior and boundary boxes. The coverage field is still
recorded because boundary boxes have fewer observations and can have greater
score variance.

## Detector training loss

The detector training path reconstructs grayscale from the normalized model
input and scales it to `[0, 1]` before passing it to the FBWR-based loss logic
in the CutLER model. That path uses `FBWRConfig.threshold_normalized`, whose
default is `0.03`.

The normalized threshold is not a replacement for the uint8 filtering
threshold. The two thresholds belong to different data representations and
must not be compared directly.

## Optimization

The image gradient magnitude is constant for every candidate bbox in one
image. The batch scoring API and COCO filtering path therefore compute it once
per image and reuse it for all annotations. The single-box APIs still compute
the gradient automatically when no precomputed gradient is supplied.

The optimization does not change the sampling geometry, score formula,
inclusive threshold comparison, or edge-sample handling.

The geometric and FBWR command-line stages display `tqdm` progress bars while
processing annotations. This is informational only and does not change the
scoring or filtering behavior.

## Calibration status

The calibration subset is external to the repository and contains 256
deterministically selected SAM 3 images. Its 4,850 raw annotations were
reduced to 4,619 after geometric refinement. With the baseline configuration
(`center_window_fraction=0.10`, `rim_margin=5`, `num_angles=12`), threshold
`2000` retained 1,316 annotations (28.5%).

On this subset, the threshold sweep retained:

| Threshold | Retained |
|---:|---:|
| 1000 | 3,231 (70.0%) |
| 1500 | 2,253 (48.8%) |
| 2000 | 1,316 (28.5%) |
| 2500 | 609 (13.2%) |
| 3000 | 195 (4.2%) |

The parameter sweep showed that `num_angles=12` is already stable: changing it
to 8 or 16 changed the number retained at threshold 2000 from 1,316 to 1,277
and 1,311, respectively. The center-window sweep had a moderate effect;
`0.05`, `0.10`, and `0.15` retained 1,345, 1,316, and 1,251 annotations.
The rim-margin sweep had a large effect on the score scale: margins of 2, 5,
8, and 12 retained 2,030, 1,316, 742, and 292 annotations at the same numeric
threshold. Therefore, `rim_margin` and `threshold` must be calibrated
together.

These results identify candidates for qualitative overlay review; they do not
by themselves select the final production configuration. The next calibration
step is to inspect retained, borderline, discarded, and boundary-touching
examples, and optionally compare them with reference annotations as an
offline diagnostic. Reference annotations must not be used to generate
pseudo-labels.

The qualitative review and offline bbox diagnostic were also run on this
subset. At IoU 0.5, the baseline configuration produced the following
precision/recall trade-off:

| Threshold | Precision | Recall |
|---:|---:|---:|
| 1000 | 0.826 | 0.277 |
| 1500 | 0.874 | 0.204 |
| 2000 | 0.897 | 0.122 |
| 2500 | 0.906 | 0.057 |

These are reference-based diagnostics only. They confirm that increasing the
threshold improves precision while reducing recall; they do not establish a
label-free threshold-selection rule. The current candidates are therefore
`1500` when retaining more plausible craters is preferred and `2000` when
prioritizing conservative pseudo-label precision. The baseline geometry
(`0.10`, `5`, `12`) remains the recommended starting geometry because angle
count is stable and rim-margin changes require a separate threshold
recalibration.
