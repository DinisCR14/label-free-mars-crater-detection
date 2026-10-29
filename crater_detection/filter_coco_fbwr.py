"""Filter Grounded SAM COCO predictions with the exact radial FBWR score."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from .fbwr import (
        FBWRConfig,
        _gradient_magnitude,
        fbwr_score_torch,
        fbwr_score_with_coverage,
    )
except ImportError:
    from fbwr import (
        FBWRConfig,
        _gradient_magnitude,
        fbwr_score_torch,
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

    for annotation in coco.get("annotations", []):
        image_info = image_by_id.get(annotation["image_id"])
        if image_info is None:
            continue
        image_id = image_info["id"]
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

        x, y, width, height = annotation["bbox"]
        box = (x, y, x + width, y + height)
        if use_torch:
            if image_id not in torch_image_cache:
                torch_image_cache[image_id] = torch.as_tensor(
                    image_cache[image_id], device=torch_device
                )
                grad_y, grad_x = torch.gradient(torch_image_cache[image_id].float())
                torch_gradient_cache[image_id] = torch.hypot(grad_x, grad_y)
            score_tensor, valid_samples, total_samples = fbwr_score_torch(
                torch_image_cache[image_id],
                box,
                config=config,
                gradients=torch_gradient_cache[image_id],
                return_coverage=True,
            )
            score = float(score_tensor.item())
        else:
            score, valid_samples, total_samples = fbwr_score_with_coverage(
                image_cache[image_id],
                box,
                config=config,
                gradients=gradient_cache[image_id],
            )
        scores.append(score)
        coverage.append(valid_samples / total_samples if total_samples else 0.0)
        if score >= threshold:
            retained_annotation = copy.deepcopy(annotation)
            retained_annotation["fbwr_score"] = score
            retained_annotation["fbwr_valid_sample_fraction"] = coverage[-1]
            kept_annotations.append(retained_annotation)
            kept_scores.append(score)

    output["annotations"] = kept_annotations
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(output, handle)

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
    args = parser.parse_args()
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
    )


if __name__ == "__main__":
    main()
