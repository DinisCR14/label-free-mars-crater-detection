"""Generate COCO instance annotations with SAM 3 concept segmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def _read_image_list(image_list: Path, dataset_root: Path) -> list[Path]:
    paths = []
    for line in image_list.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            paths.append(dataset_root / line)
    return paths


def _as_mask_array(masks: Any) -> np.ndarray:
    masks = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape [N, H, W], got {masks.shape}")
    return masks.astype(bool)


def _as_box_array(boxes: Any) -> np.ndarray:
    boxes = boxes.detach().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Expected boxes with shape [N, 4], got {boxes.shape}")
    return boxes


def _as_score_array(scores: Any) -> np.ndarray:
    scores = scores.detach().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
    return np.asarray(scores, dtype=np.float32).reshape(-1)


def _mask_annotation(
    mask: np.ndarray,
    image_id: int,
    annotation_id: int,
    score: float,
    mask_utils: Any,
) -> dict | None:
    y_indices, x_indices = np.where(mask)
    if len(x_indices) == 0:
        return None
    x_min, x_max = int(x_indices.min()), int(x_indices.max())
    y_min, y_max = int(y_indices.min()), int(y_indices.max())
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("utf-8")
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "segmentation": encoded,
        "area": int(mask.sum()),
        "bbox": [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1],
        "score": float(score),
        "iscrowd": 0,
    }


class SAM3Predictor:
    """Load SAM 3 once and run a text concept prompt on individual images."""

    def __init__(self, *, device: str = "auto"):
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        self.torch = torch
        self.device = device
        self.model = build_sam3_image_model().to(device).eval()
        self.processor = Sam3Processor(self.model)

    def predict(self, image_path: str | Path, prompt: str) -> tuple[Any, Any, Any]:
        """Return SAM 3 masks, boxes, and scores for one image."""
        image = Image.open(image_path).convert("RGB")
        state = self.processor.set_image(image)
        with self.torch.inference_mode():
            output = self.processor.set_text_prompt(state=state, prompt=prompt)
        return output["masks"], output["boxes"], output["scores"]


def generate_coco_annotations(
    image_paths: Iterable[str | Path],
    output_json: str | Path,
    predictor: SAM3Predictor,
    *,
    prompt: str,
    score_threshold: float = 0.0,
) -> None:
    """Generate raw SAM 3 concept predictions in COCO format."""
    from pycocotools import mask as mask_utils

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "crater"}],
    }
    annotation_id = 1
    for image_id, image_path in enumerate(tqdm(list(image_paths), desc="SAM 3")):
        image_path = Path(image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        height, width = image.shape[:2]
        coco["images"].append({
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height,
        })

        masks, boxes, scores = predictor.predict(image_path, prompt)
        mask_array = _as_mask_array(masks)
        box_array = _as_box_array(boxes)
        score_array = _as_score_array(scores)
        if not (len(mask_array) == len(box_array) == len(score_array)):
            raise ValueError("SAM 3 returned different numbers of masks, boxes, and scores")
        for mask, box, score in zip(mask_array, box_array, score_array):
            if score < score_threshold:
                continue
            annotation = _mask_annotation(mask, image_id, annotation_id, score, mask_utils)
            if annotation is not None:
                annotation["sam3_box"] = [float(value) for value in box]
                coco["annotations"].append(annotation)
                annotation_id += 1

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(coco, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--image-list", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--prompt", default="crater")
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    image_paths = _read_image_list(args.image_list, args.dataset_root.resolve())
    if args.limit is not None:
        image_paths = image_paths[:args.limit]
    predictor = SAM3Predictor(device=args.device)
    generate_coco_annotations(
        image_paths,
        args.output_json,
        predictor,
        prompt=args.prompt,
        score_threshold=args.score_threshold,
    )


if __name__ == "__main__":
    main()