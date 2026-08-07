"""Generate COCO annotations with Grounding DINO and Segment Anything."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


@dataclass(frozen=True)
class GroundedSAMConfig:
    grounding_model_id: str = "IDEA-Research/grounding-dino-base"
    sam_checkpoint: Path = Path("sam_vit_h.pth")
    sam_model_type: str = "vit_h"
    prompt: str = "circle"
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    device: str = "auto"

    def resolved_device(self, torch_module) -> str:
        if self.device == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        return self.device


class GroundedSAM:
    """Run Grounding DINO followed by SAM for image-level detections."""

    def __init__(self, config: GroundedSAMConfig):
        import torch
        from segment_anything import SamPredictor, sam_model_registry
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.config = config
        self.torch = torch
        self.device = config.resolved_device(torch)
        self.processor = AutoProcessor.from_pretrained(config.grounding_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            config.grounding_model_id
        ).to(self.device)
        sam = sam_model_registry[config.sam_model_type](
            checkpoint=str(config.sam_checkpoint)
        ).to(self.device)
        self.predictor = SamPredictor(sam)

    def predict(self, image_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """Return binary masks and xyxy boxes for one image."""
        image_path = Path(image_path)
        image_pil = Image.open(image_path).convert("RGB")
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        inputs = self.processor(
            images=image_pil,
            text=self.config.prompt,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.no_grad():
            outputs = self.grounding_model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
            target_sizes=[image_pil.size[::-1]],
        )
        boxes = results[0]["boxes"]
        if len(boxes) == 0:
            height, width = image_bgr.shape[:2]
            return np.empty((0, height, width), dtype=np.uint8), np.empty((0, 4), dtype=np.float32)

        transformed_boxes = self.predictor.transform.apply_boxes_torch(
            boxes.to(self.device), image_bgr.shape[:2]
        )
        self.predictor.set_image(image_bgr)
        masks, _, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        return (
            masks[:, 0].detach().cpu().numpy().astype(np.uint8),
            boxes.detach().cpu().numpy(),
        )


def _initial_coco() -> dict:
    return {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "crater"}],
    }


def _mask_annotation(mask: np.ndarray, image_id: int, annotation_id: int, mask_utils) -> dict | None:
    y_indices, x_indices = np.where(mask > 0)
    if len(x_indices) == 0:
        return None
    x_min, x_max = int(x_indices.min()), int(x_indices.max())
    y_min, y_max = int(y_indices.min()), int(y_indices.max())
    binary_mask = np.asfortranarray(mask.astype(np.uint8))
    segmentation = mask_utils.encode(binary_mask)
    segmentation["counts"] = segmentation["counts"].decode("utf-8")
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "segmentation": segmentation,
        "area": int(np.sum(mask > 0)),
        "bbox": [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1],
        "iscrowd": 0,
    }


def generate_coco_annotations(
    image_paths: Iterable[str | Path],
    output_json: str | Path,
    predictor: GroundedSAM,
) -> None:
    """Generate a COCO annotation file from an iterable of image paths."""
    from pycocotools import mask as mask_utils

    coco = _initial_coco()
    annotation_id = 1
    for image_id, image_path in enumerate(tqdm(list(image_paths), desc="Grounded SAM")):
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
        masks, _ = predictor.predict(image_path)
        for mask in masks:
            annotation = _mask_annotation(mask, image_id, annotation_id, mask_utils)
            if annotation is not None:
                coco["annotations"].append(annotation)
                annotation_id += 1

    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(coco, handle)


def _read_image_list(image_list: Path, dataset_root: Path) -> list[Path]:
    paths = []
    for line in image_list.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            paths.append(dataset_root / line)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--image-list", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--grounding-model-id", default=GroundedSAMConfig.grounding_model_id)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--sam-model-type", default=GroundedSAMConfig.sam_model_type)
    parser.add_argument("--prompt", default=GroundedSAMConfig.prompt)
    parser.add_argument("--box-threshold", type=float, default=GroundedSAMConfig.box_threshold)
    parser.add_argument("--text-threshold", type=float, default=GroundedSAMConfig.text_threshold)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=GroundedSAMConfig.device)
    args = parser.parse_args()

    config = GroundedSAMConfig(
        grounding_model_id=args.grounding_model_id,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
    )
    predictor = GroundedSAM(config)
    image_paths = _read_image_list(args.image_list, args.dataset_root)
    generate_coco_annotations(image_paths, args.output_json, predictor)


if __name__ == "__main__":
    main()
