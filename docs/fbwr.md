# FBWR design and optimization

Flexible Black and White Rims (FBWR) is used in two different stages of the
pipeline. The stages intentionally use different intensity units and
thresholds.

## Pseudo-label filtering

`crater_detection/filter_coco_fbwr.py` reads the source images as 8-bit
grayscale OpenCV arrays. It uses `FBWRConfig.threshold_uint8`, whose default is
`2000.0`, or a threshold supplied through `--threshold`.

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

## Follow-up work

The next FBWR work should add score auditability and tests for:

- CPU/Torch parity;
- threshold equality and invalid boxes;
- edge and out-of-image sampling;
- score distributions and retained/discarded counts;
- preservation of COCO metadata.
