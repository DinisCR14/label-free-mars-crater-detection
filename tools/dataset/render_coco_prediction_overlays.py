"""Render COCO prediction masks, boxes, and scores on source images."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from tqdm import tqdm


MASK_COLOR = (255, 80, 40, 90)
BOX_COLOR = (255, 220, 40)
TEXT_COLOR = (255, 255, 255)


def _decode_segmentation(segmentation: Any, size: tuple[int, int]) -> Image.Image:
    width, height = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    if isinstance(segmentation, dict):
        decoded = mask_utils.decode(segmentation)
        if decoded.ndim == 3:
            decoded = decoded[:, :, 0]
        decoded_image = Image.fromarray((decoded > 0).astype(np.uint8) * 255)
        if decoded_image.size != size:
            decoded_image = decoded_image.resize(size, Image.Resampling.NEAREST)
        return decoded_image
    if isinstance(segmentation, list):
        for polygon in segmentation:
            if len(polygon) >= 6:
                points = list(zip(polygon[::2], polygon[1::2]))
                draw.polygon(points, fill=255)
    return mask


def _render_image(
    image: Image.Image,
    annotations: list[dict[str, Any]],
) -> Image.Image:
    output = image.convert("RGBA")
    for annotation in annotations:
        segmentation = annotation.get("segmentation")
        if segmentation:
            mask = _decode_segmentation(segmentation, output.size)
            color = Image.new("RGBA", output.size, MASK_COLOR)
            output = Image.composite(color, output, mask)

    draw = ImageDraw.Draw(output)
    for annotation in annotations:
        x, y, width, height = (float(value) for value in annotation["bbox"])
        draw.rectangle((x, y, x + width, y + height), outline=BOX_COLOR, width=2)
        score = annotation.get("score")
        if score is not None:
            label = f"{float(score):.2f}"
            text_y = max(0, y - 12)
            draw.rectangle((x, text_y, x + 28, text_y + 12), fill=(0, 0, 0, 180))
            draw.text((x + 2, text_y), label, fill=TEXT_COLOR)
    return output.convert("RGB")


def _output_path(output_root: Path, file_name: str) -> Path:
    source_path = Path(file_name)
    return output_root / source_path.parent / f"{source_path.stem}_predictions.png"


def render_prediction_overlays(
    predictions_json: str | Path,
    image_root: str | Path,
    output_root: str | Path,
    *,
    score_threshold: float = 0.0,
    max_detections: int | None = None,
    limit: int | None = None,
    targets: set[str] | None = None,
) -> int:
    """Render selected COCO predictions and return the number of images written."""
    predictions = json.loads(Path(predictions_json).read_text(encoding="utf-8"))
    image_root = Path(image_root)
    output_root = Path(output_root)
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in predictions.get("annotations", []):
        if float(annotation.get("score", 0.0)) >= score_threshold:
            annotations_by_image[int(annotation["image_id"])].append(annotation)

    images = predictions.get("images", [])
    selected_images = []
    for image in images:
        file_name = str(image["file_name"])
        stem = Path(file_name).stem
        if targets and file_name not in targets and stem not in targets and str(image["id"]) not in targets:
            continue
        selected_images.append(image)
    if limit is not None:
        selected_images = selected_images[:limit]

    written = 0
    for image in tqdm(selected_images, desc="Prediction overlays"):
        image_id = int(image["id"])
        annotations = sorted(
            annotations_by_image.get(image_id, []),
            key=lambda item: float(item.get("score", 0.0)),
            reverse=True,
        )
        if max_detections is not None:
            annotations = annotations[:max_detections]
        image_path = image_root / str(image["file_name"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Could not read image: {image_path}")
        with Image.open(image_path) as source:
            rendered = _render_image(source, annotations)
        output_path = _output_path(output_root, str(image["file_name"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rendered.save(output_path)
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--max-detections", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--target", action="append", default=[])
    args = parser.parse_args()
    if not 0.0 <= args.score_threshold <= 1.0:
        parser.error("--score-threshold must be between 0 and 1")
    if args.max_detections is not None and args.max_detections < 1:
        parser.error("--max-detections must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    written = render_prediction_overlays(
        args.predictions,
        args.image_root,
        args.output,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
        limit=args.limit,
        targets=set(args.target),
    )
    print(f"Rendered {written} image overlay(s) to {args.output}")


if __name__ == "__main__":
    main()
