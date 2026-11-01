"""Measure whether a prediction filter removes false positives preferentially."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[2]))

from crater_detection.image_manifest import ImageManifest, validate_coco_mapping

SIZE_ORDER = ("small", "medium", "large")
SIZE_AREA_LIMITS = (0.0, 3.141592653589793 * 5**2, 3.141592653589793 * 20**2, 1e10)


def _load_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read COCO JSON: {path}") from error


def _as_coco_document(
    payload: dict[str, Any] | list[dict[str, Any]],
    manifest: ImageManifest,
) -> dict[str, Any]:
    if isinstance(payload, list):
        return {
            "images": [image.as_json() for image in manifest.images],
            "annotations": payload,
            "categories": [{"id": 1, "name": "crater"}],
            "mapping": manifest.metadata(),
        }
    return payload


def _validate_clipped_ground_truth(
    ground_truth: dict[str, Any], manifest: ImageManifest
) -> None:
    dimensions = {image.id: (image.width, image.height) for image in manifest.images}
    for annotation in ground_truth.get("annotations", []):
        image_id = int(annotation["image_id"])
        width, height = dimensions[image_id]
        x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
        if (
            x < 0
            or y < 0
            or box_width < 0
            or box_height < 0
            or x + box_width > width + 1e-6
            or y + box_height > height + 1e-6
        ):
            raise ValueError(
                "Ground truth must use image-clipped bounding boxes; "
                f"annotation {annotation.get('id')} exceeds image {image_id}"
            )


def _bbox_area(annotation: dict[str, Any]) -> float:
    return float(annotation["bbox"][2]) * float(annotation["bbox"][3])


def _size_for_area(area: float) -> str:
    for size, lower, upper in zip(
        SIZE_ORDER, SIZE_AREA_LIMITS, SIZE_AREA_LIMITS[1:]
    ):
        if lower <= area < upper:
            return size
    raise ValueError(f"Bounding-box area is outside configured limits: {area}")


def _bbox_iou(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_x, first_y, first_width, first_height = (
        float(value) for value in first["bbox"]
    )
    second_x, second_y, second_width, second_height = (
        float(value) for value in second["bbox"]
    )
    first_x2, first_y2 = first_x + first_width, first_y + first_height
    second_x2, second_y2 = second_x + second_width, second_y + second_height
    intersection = max(0.0, min(first_x2, second_x2) - max(first_x, second_x)) * max(
        0.0, min(first_y2, second_y2) - max(first_y, second_y)
    )
    union = _bbox_area(first) + _bbox_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _annotations_by_image(document: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for annotation in document.get("annotations", []):
        if int(annotation.get("category_id", 1)) == 1:
            grouped.setdefault(int(annotation["image_id"]), []).append(annotation)
    return grouped


def _prediction_key(annotation: dict[str, Any]) -> tuple[int, int]:
    try:
        return int(annotation["image_id"]), int(annotation["id"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Predictions must have integer image_id and id fields: {annotation!r}"
        ) from error


def _match_predictions(
    ground_truth: dict[int, list[dict[str, Any]]],
    predictions: dict[int, list[dict[str, Any]]],
    iou_threshold: float,
) -> dict[tuple[int, int], dict[str, Any]]:
    """Match predictions greedily by score and return each prediction's status."""
    matches: dict[tuple[int, int], dict[str, Any]] = {}
    for image_id in set(ground_truth) | set(predictions):
        image_ground_truth = ground_truth.get(image_id, [])
        matched_ground_truth: set[int] = set()
        ordered_predictions = sorted(
            predictions.get(image_id, []),
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        for prediction in ordered_predictions:
            key = _prediction_key(prediction)
            candidates = [
                (index, _bbox_iou(prediction, annotation))
                for index, annotation in enumerate(image_ground_truth)
                if index not in matched_ground_truth
            ]
            best_index, best_iou = max(candidates, key=lambda item: item[1]) if candidates else (-1, 0.0)
            is_true_positive = best_iou >= iou_threshold
            if is_true_positive:
                matched_ground_truth.add(best_index)
            matches[key] = {
                "true_positive": is_true_positive,
                "ground_truth_index": best_index if is_true_positive else None,
                "iou": best_iou,
            }
    return matches


def _counts(
    statuses: dict[tuple[int, int], dict[str, Any]],
    prediction_count: int,
    ground_truth_count: int,
) -> dict[str, int | float]:
    true_positives = sum(
        1 for status in statuses.values() if status["true_positive"]
    )
    false_positives = prediction_count - true_positives
    false_negatives = ground_truth_count - true_positives
    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    return {
        "predictions": prediction_count,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
    }


def _size_metrics(
    ground_truth: dict[int, list[dict[str, Any]]],
    before: dict[int, list[dict[str, Any]]],
    after_keys: set[tuple[int, int]],
    statuses: dict[tuple[int, int], dict[str, Any]],
) -> dict[str, dict[str, int | float]]:
    ground_truth_counts = {
        size: sum(
            _size_for_area(_bbox_area(annotation)) == size
            for annotations in ground_truth.values()
            for annotation in annotations
        )
        for size in SIZE_ORDER
    }
    records: dict[str, list[tuple[tuple[int, int], bool]]] = {
        size: [] for size in SIZE_ORDER
    }
    for image_id, annotations in before.items():
        image_ground_truth = ground_truth.get(image_id, [])
        for prediction in annotations:
            key = _prediction_key(prediction)
            status = statuses[key]
            size_annotation = (
                image_ground_truth[status["ground_truth_index"]]
                if status["true_positive"]
                else prediction
            )
            size = _size_for_area(_bbox_area(size_annotation))
            records[size].append((key, status["true_positive"]))

    metrics: dict[str, dict[str, int | float]] = {}
    for size in SIZE_ORDER:
        size_records = records[size]
        before_true_positives = sum(
            is_true_positive for _, is_true_positive in size_records
        )
        after_records = [
            (key, is_true_positive)
            for key, is_true_positive in size_records
            if key in after_keys
        ]
        after_true_positives = sum(
            is_true_positive for _, is_true_positive in after_records
        )
        before_predictions = len(size_records)
        after_predictions = len(after_records)
        before_false_positives = before_predictions - before_true_positives
        after_false_positives = after_predictions - after_true_positives
        before_recall = (
            before_true_positives / ground_truth_counts[size]
            if ground_truth_counts[size]
            else 0.0
        )
        after_recall = (
            after_true_positives / ground_truth_counts[size]
            if ground_truth_counts[size]
            else 0.0
        )
        metrics[size] = {
            "ground_truth": ground_truth_counts[size],
            "predictions_before": before_predictions,
            "predictions_after": after_predictions,
            "true_positives_before": before_true_positives,
            "true_positives_after": after_true_positives,
            "false_positives_before": before_false_positives,
            "false_positives_after": after_false_positives,
            "prediction_retention_rate": (
                after_predictions / before_predictions if before_predictions else 0.0
            ),
            "true_positive_retention_rate": (
                after_true_positives / before_true_positives
                if before_true_positives
                else 0.0
            ),
            "false_positive_removal_rate": (
                (before_false_positives - after_false_positives)
                / before_false_positives
                if before_false_positives
                else 0.0
            ),
            "precision_before": (
                before_true_positives / before_predictions
                if before_predictions
                else 0.0
            ),
            "precision_after": (
                after_true_positives / after_predictions
                if after_predictions
                else 0.0
            ),
            "recall_before": before_recall,
            "recall_after": after_recall,
        }
    return metrics


def analyze_filter_effect(
    ground_truth_path: str | Path,
    before_path: str | Path,
    after_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    iou_threshold: float = 0.5,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    """Analyze the TP/FP composition of predictions removed by a filter."""
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"IoU threshold must be between 0 and 1: {iou_threshold}")

    manifest = ImageManifest.load(manifest_path)
    ground_truth = _as_coco_document(_load_json(Path(ground_truth_path)), manifest)
    before = _as_coco_document(_load_json(Path(before_path)), manifest)
    after = _as_coco_document(_load_json(Path(after_path)), manifest)
    require_fingerprint = not allow_legacy
    for label, document in (
        ("ground truth", ground_truth),
        ("before predictions", before),
        ("after predictions", after),
    ):
        validate_coco_mapping(
            document,
            manifest,
            require_fingerprint=require_fingerprint,
        )
        if label == "ground truth":
            _validate_clipped_ground_truth(document, manifest)

    grouped_ground_truth = _annotations_by_image(ground_truth)
    grouped_before = _annotations_by_image(before)
    grouped_after = _annotations_by_image(after)
    before_keys = {
        _prediction_key(annotation)
        for annotations in grouped_before.values()
        for annotation in annotations
    }
    after_keys = {
        _prediction_key(annotation)
        for annotations in grouped_after.values()
        for annotation in annotations
    }
    if not after_keys <= before_keys:
        raise ValueError("After-filter predictions must be a subset of before predictions")

    statuses = _match_predictions(
        grouped_ground_truth,
        grouped_before,
        iou_threshold,
    )
    before_count = sum(len(annotations) for annotations in grouped_before.values())
    after_count = sum(len(annotations) for annotations in grouped_after.values())
    ground_truth_count = sum(len(annotations) for annotations in grouped_ground_truth.values())
    before_metrics = _counts(statuses, before_count, ground_truth_count)

    retained_statuses = {key: statuses[key] for key in after_keys}
    after_metrics = _counts(retained_statuses, after_count, ground_truth_count)
    removed_keys = before_keys - after_keys
    removed_true_positives = sum(
        1 for key in removed_keys if statuses[key]["true_positive"]
    )
    removed_false_positives = len(removed_keys) - removed_true_positives
    original_true_positives = int(before_metrics["true_positives"])
    original_false_positives = int(before_metrics["false_positives"])
    effect = {
        "removed_predictions": len(removed_keys),
        "removed_true_positives": removed_true_positives,
        "removed_false_positives": removed_false_positives,
        "false_positive_removal_rate": (
            removed_false_positives / original_false_positives
            if original_false_positives
            else 0.0
        ),
        "true_positive_retention_rate": (
            (original_true_positives - removed_true_positives) / original_true_positives
            if original_true_positives
            else 0.0
        ),
        "removed_false_positive_fraction": (
            removed_false_positives / len(removed_keys) if removed_keys else 0.0
        ),
    }
    size_metrics = _size_metrics(
        grouped_ground_truth,
        grouped_before,
        after_keys,
        statuses,
    )
    report: dict[str, Any] = {
        "ground_truth": str(ground_truth_path),
        "before": str(before_path),
        "after": str(after_path),
        "iou_threshold": iou_threshold,
        "metrics_before": before_metrics,
        "metrics_after": after_metrics,
        "filter_effect": effect,
        "size_metrics": size_metrics,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--allow-legacy", action="store_true")
    args = parser.parse_args()
    report = analyze_filter_effect(
        args.ground_truth,
        args.before,
        args.after,
        args.manifest,
        args.output,
        iou_threshold=args.iou_threshold,
        allow_legacy=args.allow_legacy,
    )
    effect = report["filter_effect"]
    print(
        f"Removed {effect['removed_predictions']} predictions: "
        f"{effect['removed_false_positives']} false positives and "
        f"{effect['removed_true_positives']} true positives"
    )
    print(
        f"False-positive removal rate: {effect['false_positive_removal_rate']:.3f}; "
        f"true-positive retention rate: {effect['true_positive_retention_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
