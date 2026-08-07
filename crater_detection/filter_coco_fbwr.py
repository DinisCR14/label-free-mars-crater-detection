"""Filter Grounded SAM COCO predictions with the exact radial FBWR score."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np

try:
    from .fbwr import FBWRConfig, fbwr_score, fbwr_score_torch
except ImportError:
    from fbwr import FBWRConfig, fbwr_score, fbwr_score_torch

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
) -> None:
    """Write a COCO prediction file containing only annotations above threshold."""
    input_json = Path(input_json)
    image_root = Path(image_root)
    output_json = Path(output_json)
    with input_json.open("r", encoding="utf-8") as handle:
        coco = json.load(handle)

    output = copy.deepcopy(coco)
    image_by_id = {image["id"]: image for image in coco.get("images", [])}
    kept_annotations = []
    config = FBWRConfig()
    image_cache: dict[int, np.ndarray] = {}
    use_torch = torch is not None and (
        device == "cuda" or (device == "auto" and torch.cuda.is_available())
    )
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
            image_cache[image_id] = image

        x, y, width, height = annotation["bbox"]
        box = (x, y, x + width, y + height)
        if use_torch:
            image_tensor = torch.as_tensor(image_cache[image_id], device=torch_device)
            score = float(fbwr_score_torch(image_tensor, box, config=config).item())
        else:
            score = fbwr_score(image_cache[image_id], box, config=config)
        if score > threshold:
            kept_annotations.append(annotation)

    output["annotations"] = kept_annotations
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(output, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--threshold", type=float, default=2000.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    filter_coco_predictions(
        args.input_json,
        args.image_root,
        args.output_json,
        threshold=args.threshold,
        device=args.device,
    )


if __name__ == "__main__":
    main()
