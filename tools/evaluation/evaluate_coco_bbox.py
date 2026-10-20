"""Evaluate COCO bounding-box predictions with the standard COCO protocol."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2]))

from crater_detection.image_manifest import ImageManifest, validate_coco_mapping


COCO_STAT_NAMES = (
    "AP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR_1",
    "AR_10",
    "AR_100",
    "AR_small",
    "AR_medium",
    "AR_large",
)
CUSTOM_AREA_RANGES = (
    ("small", [0.0, 3.141592653589793 * 5**2]),
    ("medium", [3.141592653589793 * 5**2, 3.141592653589793 * 20**2]),
    ("large", [3.141592653589793 * 20**2, 1e10]),
)
SIZE_ORDER = ("small", "medium", "large")


def _load_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read COCO JSON: {path}") from error


def _as_coco_document(
    payload: dict[str, Any] | list[dict[str, Any]],
    manifest: ImageManifest,
) -> dict[str, Any]:
    """Normalize a COCO object or Detectron2 prediction list."""
    if isinstance(payload, list):
        return {
            "images": [image.as_json() for image in manifest.images],
            "annotations": payload,
            "categories": [{"id": 1, "name": "crater"}],
            "mapping": manifest.metadata(),
        }
    return payload


def _validate_predictions(
    payload: dict[str, Any],
    manifest: ImageManifest,
    *,
    require_fingerprint: bool,
) -> None:
    validate_coco_mapping(
        payload,
        manifest,
        require_fingerprint=require_fingerprint,
    )
    image_ids = {image.id for image in manifest.images}
    for annotation in payload.get("annotations", []):
        try:
            image_id = int(annotation["image_id"])
            bbox = annotation["bbox"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Invalid prediction annotation: {annotation!r}") from error
        if image_id not in image_ids:
            raise ValueError(f"Prediction references unknown image ID: {image_id}")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Prediction bbox must have four values: {annotation!r}")
        if any(float(value) < 0 for value in (bbox[2], bbox[3])):
            raise ValueError(f"Prediction bbox has negative size: {annotation!r}")


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


def _with_bbox_areas(document: dict[str, Any]) -> dict[str, Any]:
    """Use bbox area for COCO area filtering without changing the source document."""
    normalized = copy.deepcopy(document)
    for annotation in normalized.get("annotations", []):
        annotation["area"] = float(annotation["bbox"][2]) * float(annotation["bbox"][3])
    return normalized


def _run_evaluation(
    ground_truth: dict[str, Any],
    predictions: dict[str, Any],
    *,
    area_ranges: tuple[tuple[str, list[float]], ...] | None = None,
) -> Any:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = ground_truth
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(predictions.get("annotations", []))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    if area_ranges is not None:
        evaluator.params.areaRng = [[0.0, 1e10], *[bounds for _, bounds in area_ranges]]
        evaluator.params.areaRngLbl = ["all", *[name for name, _ in area_ranges]]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return evaluator


def evaluate_model(
    ground_truth: dict[str, Any],
    predictions: dict[str, Any],
    *,
    area_ranges: tuple[tuple[str, list[float]], ...] | None = None,
) -> dict[str, float]:
    """Run COCO bbox evaluation and return the twelve standard summary values."""
    evaluator = _run_evaluation(
        ground_truth,
        predictions,
        area_ranges=area_ranges,
    )
    return {
        name: float(value)
        for name, value in zip(COCO_STAT_NAMES, evaluator.stats)
    }


def _evaluate_with_custom_areas(
    ground_truth: dict[str, Any], predictions: dict[str, Any]
) -> dict[str, float]:
    return evaluate_model(
        _with_bbox_areas(ground_truth),
        _with_bbox_areas(predictions),
        area_ranges=CUSTOM_AREA_RANGES,
    )


def _plot_precision_recall(
    curve_data: dict[str, dict[str, Any]], output_path: Path
) -> None:
    rows = ("all", "small", "medium", "large")
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    for index, area in enumerate(rows):
        axis = axes[index // 2][index % 2]
        for label, data in curve_data.items():
            precision = data["precision"]
            iou_index = 0
            if precision.shape[0] > 1:
                iou_index = min(5, precision.shape[0] - 1)
            area_index = data["area_labels"].index(area)
            values = precision[iou_index, :, 0, area_index, 2]
            valid = values > -1
            axis.plot(
                data["recall"][valid],
                values[valid],
                label=label,
            )
        axis.set_title(f"COCO precision-recall ({area})")
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.05)
        axis.grid(True)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_ap_vs_iou(
    curve_data: dict[str, dict[str, Any]], output_path: Path
) -> None:
    rows = ("all", "small", "medium", "large")
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)
    for index, area in enumerate(rows):
        axis = axes[index // 2][index % 2]
        for label, data in curve_data.items():
            area_index = data["area_labels"].index(area)
            values = data["precision"][:, :, 0, area_index, 2]
            ap = [
                float(values[row][values[row] > -1].mean())
                if (values[row] > -1).any()
                else 0.0
                for row in range(values.shape[0])
            ]
            axis.plot(data["iou_thresholds"], ap, ".-", label=label)
        axis.set_title(f"AP versus IoU ({area})")
        axis.set_xlabel("IoU threshold")
        axis.set_ylabel("AP")
        axis.set_ylim(0, 1.05)
        axis.grid(True)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_ap_by_size(
    evaluated: dict[str, dict[str, dict[str, float]]], output_path: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)
    for index, protocol in enumerate(("standard_coco", "custom_bbox_area")):
        axis = axes[0][index]
        positions = np.arange(len(SIZE_ORDER))
        width = 0.8 / max(1, len(evaluated))
        for model_index, (label, values) in enumerate(evaluated.items()):
            heights = [values[protocol][f"AP_{size}"] for size in SIZE_ORDER]
            axis.bar(positions + model_index * width, heights, width, label=label)
        axis.set_title(f"AP by size ({protocol.replace('_', ' ')})")
        axis.set_xticks(positions + width * (len(evaluated) - 1) / 2)
        axis.set_xticklabels(SIZE_ORDER)
        axis.set_ylim(0, 1.05)
        axis.set_ylabel("AP@[0.50:0.95]")
        axis.grid(axis="y")
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def evaluate_coco_bbox(
    ground_truth_path: str | Path,
    models: list[tuple[str, str | Path]],
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    """Validate and evaluate one or more COCO bbox prediction files."""
    manifest = ImageManifest.load(manifest_path)
    ground_truth_path = Path(ground_truth_path)
    ground_truth = _as_coco_document(_load_json(ground_truth_path), manifest)
    validate_coco_mapping(
        ground_truth,
        manifest,
        require_fingerprint=not allow_legacy,
    )
    if ground_truth.get("annotations") is None:
        raise ValueError("Ground truth must contain annotations")
    _validate_clipped_ground_truth(ground_truth, manifest)

    evaluated: dict[str, dict[str, dict[str, float]]] = {}
    curve_data: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    for label, path_value in models:
        path = Path(path_value)
        predictions = _as_coco_document(_load_json(path), manifest)
        _validate_predictions(
            predictions,
            manifest,
            require_fingerprint=not allow_legacy,
        )
        standard_evaluator = _run_evaluation(ground_truth, predictions)
        custom_evaluator = _run_evaluation(
            _with_bbox_areas(ground_truth),
            _with_bbox_areas(predictions),
            area_ranges=CUSTOM_AREA_RANGES,
        )
        evaluated[label] = {
            "standard_coco": {
                name: float(value)
                for name, value in zip(COCO_STAT_NAMES, standard_evaluator.stats)
            },
            "custom_bbox_area": {
                name: float(value)
                for name, value in zip(COCO_STAT_NAMES, custom_evaluator.stats)
            },
        }
        curve_data[label] = {
            "precision": custom_evaluator.eval["precision"],
            "recall": custom_evaluator.params.recThrs,
            "iou_thresholds": custom_evaluator.params.iouThrs,
            "area_labels": custom_evaluator.params.areaRngLbl,
        }
        sources[label] = str(path)

    report = {
        "protocol": {
            "evaluator": "pycocotools.COCOeval",
            "iou_type": "bbox",
            "ground_truth": str(ground_truth_path),
            "ground_truth_mapping": ground_truth.get("mapping"),
            "manifest": str(Path(manifest_path)),
            "manifest_fingerprint": manifest.fingerprint,
            "image_count": len(manifest.images),
            "allow_legacy": allow_legacy,
            "custom_area_ranges": {
                name: bounds for name, bounds in CUSTOM_AREA_RANGES
            },
        },
        "models": evaluated,
        "sources": sources,
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "coco_bbox_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    _plot_precision_recall(curve_data, output_dir / "coco_precision_recall.png")
    _plot_ap_vs_iou(curve_data, output_dir / "coco_ap_vs_iou.png")
    _plot_ap_by_size(evaluated, output_dir / "coco_ap_by_size.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, metavar="LABEL=JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-legacy",
        action="store_true",
        help="allow COCO files without manifest metadata while checking image records",
    )
    args = parser.parse_args()
    models = []
    for specification in args.model:
        label, separator, path = specification.partition("=")
        if not separator or not label or not path:
            parser.error(f"--model must use LABEL=JSON: {specification}")
        models.append((label, path))
    try:
        evaluate_coco_bbox(
            args.ground_truth,
            models,
            args.manifest,
            args.output_dir,
            allow_legacy=args.allow_legacy,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
