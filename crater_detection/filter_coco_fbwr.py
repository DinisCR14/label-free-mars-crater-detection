"""Filter Grounded SAM COCO predictions with the exact radial FBWR score."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    from .fbwr import (
        FBWRConfig,
        _gradient_magnitude,
        fbwr_scores_torch,
        fbwr_score_with_coverage,
    )
except ImportError:
    from fbwr import (
        FBWRConfig,
        _gradient_magnitude,
        fbwr_scores_torch,
        fbwr_score_with_coverage,
    )

try:
    import torch
except ImportError:
    torch = None


def filter_coco_predictions(
    input_json: str | Path,
    image_root: str | Path,
    output_json: str | Path,
    *,
    threshold: float = 2000.0,
    device: str = "auto",
    report_json: str | Path | None = None,
    center_window_fraction: float = 0.10,
    rim_margin: int = 5,
    num_angles: int = 12,
    scores_json: str | Path | None = None,
) -> None:
    """Write FBWR-retained COCO predictions and a score audit report."""
    input_json = Path(input_json)
    image_root = Path(image_root)
    output_json = Path(output_json)
    if report_json is None:
        report_json = output_json.with_name(f"{output_json.stem}_report.json")
    report_json = Path(report_json)
    with input_json.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    output = copy.deepcopy(coco)
    image_by_id = {image["id"]: image for image in coco.get("images", [])}
    kept_annotations = []
    scored_annotations = []
    config = FBWRConfig(
        center_window_fraction=center_window_fraction,
        rim_margin=rim_margin,
        num_angles=num_angles,
    )
    image_cache: dict[int, np.ndarray] = {}
    gradient_cache: dict[int, np.ndarray] = {}
    torch_image_cache = {}
    torch_gradient_cache = {}
    scores: list[float] = []
    kept_scores: list[float] = []
    coverage: list[float] = []
    use_torch = torch is not None and (
        device == "cuda" or (device == "auto" and torch.cuda.is_available())
    )
    if device == "cuda" and (torch is None or not torch.cuda.is_available()):
        raise RuntimeError("--device cuda requires an available CUDA-enabled PyTorch")
    torch_device = torch.device("cuda" if device == "auto" else device) if use_torch else None

    annotations_by_image: dict[int, list[dict]] = {}
    for annotation in coco.get("annotations", []):
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    for image_id, image_annotations in tqdm(
        annotations_by_image.items(),
        desc="FBWR filtering",
        unit="image",
    ):
        image_info = image_by_id.get(image_id)
        if image_info is None:
            continue
        if image_id not in image_cache:
            image_path = image_root / image_info["file_name"]
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")
            expected_size = (image_info.get("width"), image_info.get("height"))
            if all(value is not None for value in expected_size):
                actual_size = (image.shape[1], image.shape[0])
                if actual_size != expected_size:
                    raise ValueError(
                        f"Image dimensions for {image_path} are {actual_size}, "
                        f"expected {expected_size} from COCO metadata"
                    )
            image_cache[image_id] = image
            if not use_torch:
                gradient_cache[image_id] = _gradient_magnitude(image)

        if use_torch:
            if image_id not in torch_image_cache:
                torch_image_cache[image_id] = torch.as_tensor(
                    image_cache[image_id], device=torch_device
                )
                grad_y, grad_x = torch.gradient(torch_image_cache[image_id].float())
                torch_gradient_cache[image_id] = torch.hypot(grad_x, grad_y)
            boxes = [
                (x, y, x + width, y + height)
                for x, y, width, height in (annotation["bbox"] for annotation in image_annotations)
            ]
            score_tensors, valid_samples_list, total_samples_list = fbwr_scores_torch(
                torch_image_cache[image_id],
                boxes,
                config=config,
                gradients=torch_gradient_cache[image_id],
                return_coverage=True,
            )
        else:
            scores_list = []
            valid_samples_list = []
            total_samples_list = []
            for annotation in image_annotations:
                x, y, width, height = annotation["bbox"]
                score, valid_samples, total_samples = fbwr_score_with_coverage(
                    image_cache[image_id], (x, y, x + width, y + height),
                    config=config, gradients=gradient_cache[image_id],
                )
                scores_list.append(score)
                valid_samples_list.append(valid_samples)
                total_samples_list.append(total_samples)
        for annotation, score, valid_samples, total_samples in zip(
            image_annotations,
            scores_list if not use_torch else score_tensors.detach().cpu().tolist(),
            valid_samples_list,
            total_samples_list,
        ):
            score = float(score)
            scores.append(score)
            coverage_value = valid_samples / total_samples if total_samples else 0.0
            coverage.append(coverage_value)
            scored_annotation = copy.deepcopy(annotation)
            scored_annotation["fbwr_score"] = score
            scored_annotation["fbwr_valid_sample_fraction"] = coverage_value
            scored_annotations.append(scored_annotation)
            if score >= threshold:
                kept_annotations.append(scored_annotation)
                kept_scores.append(score)

    output["annotations"] = kept_annotations
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(output, handle)
    if scores_json is not None:
        scored_output = copy.deepcopy(coco)
        scored_output["annotations"] = scored_annotations
        scored_output["fbwr"] = {
            "center_window_fraction": config.center_window_fraction,
            "rim_margin": config.rim_margin,
            "num_angles": config.num_angles,
            "device": "torch" if use_torch else "cpu",
        }
        scores_path = Path(scores_json)
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        with scores_path.open("w", encoding="utf-8") as handle:
            json.dump(scored_output, handle)

    def summary(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"min": None, "max": None, "mean": None, "median": None}
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "median": float(np.median(values)),
        }

    report = {
        "input_json": str(input_json),
        "image_root": str(image_root),
        "output_json": str(output_json),
        "threshold": threshold,
        "comparison": ">=",
        "config": {
            "center_window_fraction": config.center_window_fraction,
            "rim_margin": config.rim_margin,
            "num_angles": config.num_angles,
        },
        "device": "torch" if use_torch else "cpu",
        "input_annotation_count": len(scores),
        "retained_annotation_count": len(kept_annotations),
        "discarded_annotation_count": len(scores) - len(kept_annotations),
        "all_score_summary": summary(scores),
        "retained_score_summary": summary(kept_scores),
        "valid_sample_fraction_summary": summary(coverage),
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    with report_json.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def filter_scored_predictions(
    scores_json: str | Path,
    output_json: str | Path,
    *,
    threshold: float = 2000.0,
    report_json: str | Path | None = None,
    expected_config: dict[str, float | int] | None = None,
    input_json: str | Path | None = None,
) -> None:
    """Filter a previously scored COCO document without recomputing FBWR."""
    scores_path = Path(scores_json)
    output_path = Path(output_json)
    if report_json is None:
        report_json = output_path.with_name(f"{output_path.stem}_report.json")
    scored = json.loads(scores_path.read_text(encoding="utf-8"))
    stored_config = scored.get("fbwr")
    if not isinstance(stored_config, dict):
        raise ValueError(f"FBWR score cache has no configuration: {scores_path}")
    config_keys = ("center_window_fraction", "rim_margin", "num_angles")
    if expected_config is not None:
        for key in config_keys:
            if stored_config.get(key) != expected_config[key]:
                raise ValueError(
                    f"FBWR score cache {scores_path} has incompatible {key}: "
                    f"{stored_config.get(key)!r} != {expected_config[key]!r}"
                )
    if input_json is not None:
        source = json.loads(Path(input_json).read_text(encoding="utf-8"))
        source_keys = {
            (int(annotation["image_id"]), int(annotation["id"]))
            for annotation in source.get("annotations", [])
        }
        scored_keys = {
            (int(annotation["image_id"]), int(annotation["id"]))
            for annotation in scored.get("annotations", [])
        }
        if source_keys != scored_keys:
            raise ValueError("FBWR score cache does not match refined predictions")

    annotations = scored.get("annotations", [])
    for annotation in annotations:
        if "fbwr_score" not in annotation or "fbwr_valid_sample_fraction" not in annotation:
            raise ValueError(f"FBWR score cache annotation is missing scores: {annotation!r}")
    kept = [
        copy.deepcopy(annotation)
        for annotation in annotations
        if float(annotation["fbwr_score"]) >= threshold
    ]
    output = copy.deepcopy(scored)
    output.pop("fbwr", None)
    output["annotations"] = kept
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output), encoding="utf-8")

    all_scores = [float(annotation["fbwr_score"]) for annotation in annotations]
    kept_scores = [float(annotation["fbwr_score"]) for annotation in kept]

    def summary(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"min": None, "max": None, "mean": None, "median": None}
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "median": float(np.median(values)),
        }

    report = {
        "input_json": str(scores_path),
        "output_json": str(output_path),
        "threshold": threshold,
        "comparison": ">=",
        "config": {key: stored_config[key] for key in config_keys},
        "device": stored_config.get("device", "unknown"),
        "input_annotation_count": len(annotations),
        "retained_annotation_count": len(kept),
        "discarded_annotation_count": len(annotations) - len(kept),
        "all_score_summary": summary(all_scores),
        "retained_score_summary": summary(kept_scores),
        "valid_sample_fraction_summary": summary(
            [float(annotation["fbwr_valid_sample_fraction"]) for annotation in annotations]
        ),
        "score_cache": True,
    }
    report_path = Path(report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--threshold", type=float, default=2000.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--center-window-fraction", type=float, default=0.10)
    parser.add_argument("--rim-margin", type=int, default=5)
    parser.add_argument("--num-angles", type=int, default=12)
    parser.add_argument(
        "--scores-json",
        type=Path,
        help="Also write all scored annotations for later threshold filtering",
    )
    parser.add_argument(
        "--existing-scores",
        type=Path,
        help="Filter an existing score cache instead of recomputing FBWR",
    )
    args = parser.parse_args()
    if args.existing_scores is not None and args.scores_json is not None:
        parser.error("--existing-scores and --scores-json cannot be combined")
    config = {
        "center_window_fraction": args.center_window_fraction,
        "rim_margin": args.rim_margin,
        "num_angles": args.num_angles,
    }
    if args.existing_scores is not None:
        filter_scored_predictions(
            args.existing_scores,
            args.output_json,
            threshold=args.threshold,
            report_json=args.report_json,
            expected_config=config,
            input_json=args.input_json,
        )
    else:
        filter_coco_predictions(
            args.input_json,
            args.image_root,
            args.output_json,
            threshold=args.threshold,
            device=args.device,
            report_json=args.report_json,
            center_window_fraction=args.center_window_fraction,
            rim_margin=args.rim_margin,
            num_angles=args.num_angles,
            scores_json=args.scores_json,
        )


if __name__ == "__main__":
    main()
