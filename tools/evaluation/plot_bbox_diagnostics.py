"""Plot supplementary score-ranked greedy bbox diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[2]))

from crater_detection.image_manifest import ImageManifest, validate_coco_mapping
from tools.evaluation.evaluate_coco_bbox import (
    _as_coco_document,
    _load_json,
    _validate_clipped_ground_truth,
)


SIZE_ORDER = ("small", "medium", "large")
# Add "detection_jaccard" here if that supplementary metric is needed again.
METRIC_ORDER = ("precision", "recall", "f1")
AREA_LIMITS = (0.0, float(np.pi * 5**2), float(np.pi * 20**2), 1e10)


def _annotations_by_image(data: dict) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for annotation in data.get("annotations", []):
        if int(annotation.get("category_id", 1)) != 1:
            continue
        grouped.setdefault(int(annotation["image_id"]), []).append(annotation)
    return grouped


def _bbox_area(annotation: dict) -> float:
    return float(annotation["bbox"][2]) * float(annotation["bbox"][3])


def _size_for_area(area: float) -> str:
    for size, lower, upper in zip(SIZE_ORDER, AREA_LIMITS, AREA_LIMITS[1:]):
        if lower <= area < upper:
            return size
    raise ValueError(f"Bounding-box area is outside configured limits: {area}")


def _bbox_iou(first: dict, second: dict) -> float:
    first_x, first_y, first_width, first_height = first["bbox"]
    second_x, second_y, second_width, second_height = second["bbox"]
    first_x2, first_y2 = first_x + first_width, first_y + first_height
    second_x2, second_y2 = second_x + second_width, second_y + second_height
    intersection = max(0.0, min(first_x2, second_x2) - max(first_x, second_x)) * max(
        0.0, min(first_y2, second_y2) - max(first_y, second_y)
    )
    union = _bbox_area(first) + _bbox_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _match_counts(
    all_ground_truth: Iterable[dict],
    predictions: Iterable[dict],
    iou_threshold: float,
    max_detections: int,
    size: str | None = None,
) -> tuple[int, int, int]:
    all_ground_truth = list(all_ground_truth)
    in_range = (
        all_ground_truth
        if size is None
        else [
            annotation
            for annotation in all_ground_truth
            if _size_for_area(_bbox_area(annotation)) == size
        ]
    )
    out_of_range = [
        annotation for annotation in all_ground_truth if annotation not in in_range
    ]
    predictions = sorted(
        predictions, key=lambda item: float(item.get("score", 0.0)), reverse=True
    )[:max_detections]
    matched_ground_truth: set[int] = set()
    true_positive = 0
    false_positive = 0
    for prediction in predictions:
        candidates = [
            (index, _bbox_iou(prediction, annotation))
            for index, annotation in enumerate(in_range)
            if index not in matched_ground_truth
        ]
        if candidates:
            best_index, best_iou = max(candidates, key=lambda item: item[1])
            if best_iou >= iou_threshold:
                matched_ground_truth.add(best_index)
                true_positive += 1
                continue
        if any(_bbox_iou(prediction, annotation) >= iou_threshold for annotation in out_of_range):
            continue
        false_positive += 1
    false_negative = len(in_range) - true_positive
    return true_positive, false_positive, false_negative


def _metric_values(
    true_positive: int, false_positive: int, false_negative: int
) -> dict[str, float]:
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # detection_jaccard = (
    #     true_positive / (true_positive + false_positive + false_negative)
    #     if true_positive + false_positive + false_negative
    #     else 0.0
    # )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        # "detection_jaccard": detection_jaccard,
    }


def evaluate_model(
    ground_truth: dict[int, list[dict]],
    predictions: dict[int, list[dict]],
    iou_thresholds: Iterable[float],
    *,
    size: str | None = None,
    max_detections: int = 100,
) -> dict[str, list[float]]:
    values = {metric: [] for metric in METRIC_ORDER}
    image_ids = set(ground_truth) | set(predictions)
    for iou_threshold in tqdm(
        iou_thresholds, desc=f"Diagnostics ({size or 'all'})", leave=False
    ):
        counts = [0, 0, 0]
        for image_id in image_ids:
            matched = _match_counts(
                ground_truth.get(image_id, []),
                predictions.get(image_id, []),
                iou_threshold,
                max_detections,
                size=size,
            )
            counts = [left + right for left, right in zip(counts, matched)]
        metrics = _metric_values(*counts)
        for metric, value in metrics.items():
            values[metric].append(value)
    return values


def _plot(
    evaluated: dict[str, dict[str, dict[str, list[float]]]],
    thresholds: list[float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for sizes, filename, title in (
        (("all",), "greedy_bbox_metrics_all.png", "All detections"),
        (SIZE_ORDER, "greedy_bbox_metrics_by_size.png", "By ground-truth size"),
    ):
        figure, axes = plt.subplots(
            len(sizes),
            len(METRIC_ORDER),
            squeeze=False,
            figsize=(15, 4 * len(sizes)),
        )
        for row, size in enumerate(sizes):
            for column, metric in enumerate(METRIC_ORDER):
                axis = axes[row, column]
                for label, model_values in evaluated.items():
                    axis.plot(
                        thresholds,
                        model_values[size][metric],
                        ".-",
                        linewidth=2,
                        label=label,
                    )
                axis.set_title(f"{metric.replace('_', ' ').title()} ({size})")
                axis.set_xlabel("IoU threshold")
                axis.set_ylabel(metric.replace("_", " ").title())
                axis.set_ylim(0, 1.05)
                axis.grid(True)
                axis.legend()
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=200)
        plt.close(figure)


def plot_bbox_diagnostics(
    ground_truth_path: str | Path,
    models: Iterable[tuple[str, str | Path]],
    output_dir: str | Path,
    manifest_path: str | Path,
    *,
    iou_thresholds: Iterable[float] = tuple(np.arange(0.1, 1.0, 0.1)),
    max_detections: int = 100,
    allow_legacy: bool = False,
) -> dict:
    manifest = ImageManifest.load(manifest_path)
    ground_truth = _as_coco_document(_load_json(Path(ground_truth_path)), manifest)
    validate_coco_mapping(
        ground_truth, manifest, require_fingerprint=not allow_legacy
    )
    _validate_clipped_ground_truth(ground_truth, manifest)
    thresholds = [float(value) for value in iou_thresholds]
    grouped_ground_truth = _annotations_by_image(ground_truth)
    evaluated = {}
    sources = {}
    for label, path_value in models:
        path = Path(path_value)
        predictions = _as_coco_document(_load_json(path), manifest)
        validate_coco_mapping(
            predictions, manifest, require_fingerprint=not allow_legacy
        )
        grouped_predictions = _annotations_by_image(predictions)
        evaluated[label] = {
            "all": evaluate_model(
                grouped_ground_truth,
                grouped_predictions,
                thresholds,
                max_detections=max_detections,
            )
        }
        for size in SIZE_ORDER:
            evaluated[label][size] = evaluate_model(
                grouped_ground_truth,
                grouped_predictions,
                thresholds,
                size=size,
                max_detections=max_detections,
            )
        sources[label] = str(path)

    report = {
        "protocol": {
            "evaluator": "custom_score_ranked_greedy",
            "ground_truth_size_assignment": "bbox_area",
            "prediction_size_filtering": "none",
            "out_of_range_ground_truth_matches": "ignored",
            "iou_thresholds": thresholds,
            "max_detections": max_detections,
            "manifest_fingerprint": manifest.fingerprint,
            "image_count": len(manifest.images),
            "allow_legacy": allow_legacy,
            "area_ranges": {
                size: [lower, upper]
                for size, lower, upper in zip(
                    SIZE_ORDER, AREA_LIMITS, AREA_LIMITS[1:]
                )
            },
        },
        "metrics": evaluated,
        "sources": sources,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "greedy_bbox_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _plot(evaluated, thresholds, output_dir)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, metavar="LABEL=JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--iou-thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--allow-legacy", action="store_true")
    args = parser.parse_args()
    if args.max_detections < 1:
        parser.error("--max-detections must be positive")
    models = []
    for specification in args.model:
        label, separator, path = specification.partition("=")
        if not separator or not label or not path:
            parser.error(f"--model must use LABEL=JSON: {specification}")
        models.append((label, path))
    thresholds = [float(value) for value in args.iou_thresholds.split(",") if value.strip()]
    try:
        plot_bbox_diagnostics(
            args.ground_truth,
            models,
            args.output_dir,
            args.manifest,
            iou_thresholds=thresholds,
            max_detections=args.max_detections,
            allow_legacy=args.allow_legacy,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
