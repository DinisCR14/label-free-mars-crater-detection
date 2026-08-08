"""Plot COCO-style bounding-box metrics over IoU thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


SIZE_ORDER = ("small", "medium", "large")
METRIC_ORDER = ("precision", "recall", "f1", "accuracy")
AREA_LIMITS = (0.0, float(np.pi * 5**2), float(np.pi * 20**2), 1e10)


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _annotations_by_image(data) -> dict[int, list[dict]]:
    annotations = data.get("annotations", []) if isinstance(data, dict) else data
    grouped: dict[int, list[dict]] = {}
    for annotation in annotations:
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
    intersection_width = max(0.0, min(first_x2, second_x2) - max(first_x, second_x))
    intersection_height = max(0.0, min(first_y2, second_y2) - max(first_y, second_y))
    intersection = intersection_width * intersection_height
    union = _bbox_area(first) + _bbox_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _match_counts(
    ground_truth: Iterable[dict],
    predictions: Iterable[dict],
    iou_threshold: float,
    max_detections: int,
) -> tuple[int, int, int]:
    ground_truth = list(ground_truth)
    predictions = sorted(predictions, key=lambda item: float(item.get("score", 0.0)), reverse=True)
    predictions = predictions[:max_detections]
    matched_ground_truth: set[int] = set()
    true_positive = 0
    for prediction in predictions:
        candidates = [
            (index, _bbox_iou(prediction, annotation))
            for index, annotation in enumerate(ground_truth)
            if index not in matched_ground_truth
        ]
        if candidates:
            best_index, best_iou = max(candidates, key=lambda item: item[1])
            if best_iou >= iou_threshold:
                matched_ground_truth.add(best_index)
                true_positive += 1
    false_positive = len(predictions) - true_positive
    false_negative = len(ground_truth) - true_positive
    return true_positive, false_positive, false_negative


def _metric_values(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float]:
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = true_positive / (true_positive + false_positive + false_negative) if true_positive + false_positive + false_negative else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


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
    for iou_threshold in iou_thresholds:
        counts = [0, 0, 0]
        for image_id in image_ids:
            image_ground_truth = ground_truth.get(image_id, [])
            image_predictions = predictions.get(image_id, [])
            if size is not None:
                image_ground_truth = [
                    annotation for annotation in image_ground_truth
                    if _size_for_area(_bbox_area(annotation)) == size
                ]
                image_predictions = [
                    annotation for annotation in image_predictions
                    if _size_for_area(_bbox_area(annotation)) == size
                ]
            matched = _match_counts(
                image_ground_truth, image_predictions, iou_threshold, max_detections
            )
            counts = [left + right for left, right in zip(counts, matched)]
        metrics = _metric_values(*counts)
        for metric, value in metrics.items():
            values[metric].append(value)
    return values


def _plot(
    evaluated: dict[str, dict[str, dict[str, list[float]]]],
    iou_thresholds: list[float],
    *,
    sizes: tuple[str, ...] | None,
    output_path: Path,
) -> None:
    row_labels = sizes or (None,)
    figure, axes = plt.subplots(
        len(row_labels), len(METRIC_ORDER), squeeze=False,
        figsize=(16, 4 * len(row_labels)),
    )
    for row, size in enumerate(row_labels):
        for column, metric in enumerate(METRIC_ORDER):
            axis = axes[row, column]
            for label, model_values in evaluated.items():
                key = size or "all"
                axis.plot(
                    iou_thresholds, model_values[key][metric], ".-",
                    linewidth=2, label=label,
                )
            title = metric.replace("f1", "F1").title()
            axis.set_title(f"{title}{f' ({size})' if size else ''}")
            axis.set_xlabel("IoU Threshold")
            axis.set_ylabel("F1 Score" if metric == "f1" else metric.title())
            axis.set_ylim(0, 1.05)
            axis.grid(True)
            axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_bbox_metrics(
    ground_truth_path: str | Path,
    models: Iterable[tuple[str, str | Path]],
    output_dir: str | Path,
    *,
    iou_thresholds: Iterable[float] = tuple(np.arange(0.1, 1.0, 0.1)),
    max_detections: int = 100,
) -> dict:
    thresholds = [float(value) for value in iou_thresholds]
    ground_truth = _annotations_by_image(_load_json(Path(ground_truth_path)))
    evaluated: dict[str, dict[str, dict[str, list[float]]]] = {}
    for label, prediction_path in models:
        predictions = _annotations_by_image(_load_json(Path(prediction_path)))
        evaluated[label] = {
            "all": evaluate_model(
                ground_truth, predictions, thresholds,
                max_detections=max_detections,
            )
        }
        for size in SIZE_ORDER:
            evaluated[label][size] = evaluate_model(
                ground_truth, predictions, thresholds,
                size=size, max_detections=max_detections,
            )

    output_dir = Path(output_dir)
    _plot(evaluated, thresholds, sizes=None, output_path=output_dir / "bbox_metrics_all.png")
    _plot(
        evaluated, thresholds, sizes=SIZE_ORDER,
        output_path=output_dir / "bbox_metrics_by_size.png",
    )
    with (output_dir / "bbox_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"iou_thresholds": thresholds, "metrics": evaluated}, handle, indent=2)
    return evaluated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, metavar="LABEL=JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--iou-thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    args = parser.parse_args()
    models = []
    for specification in args.model:
        label, separator, path = specification.partition("=")
        if not separator or not label or not path:
            parser.error(f"--model must use LABEL=JSON: {specification}")
        models.append((label, path))
    thresholds = [float(value) for value in args.iou_thresholds.split(",") if value.strip()]
    plot_bbox_metrics(
        args.ground_truth, models, args.output_dir,
        iou_thresholds=thresholds, max_detections=args.max_detections,
    )


if __name__ == "__main__":
    main()