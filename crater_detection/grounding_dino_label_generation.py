"""Generate COCO bounding-box annotations with Grounding DINO only."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from tqdm import tqdm


DEFAULT_MODEL_ID = "IDEA-Research/grounding-dino-base"
DEFAULT_PROMPT = "circle"
DEFAULT_BOX_THRESHOLD = 0.30
DEFAULT_TEXT_THRESHOLD = 0.25


def _discover_images(image_root: Path) -> list[Path]:
    return sorted(image_root.glob("*_corrected.png"))


def _read_image_list(image_list: Path, dataset_root: Path) -> list[Path]:
    paths = []
    for line in image_list.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            paths.append(dataset_root / line)
    return paths


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_prediction(path: Path, image_id: int, file_name: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        prediction = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    image = prediction.get("image", {})
    if image.get("id") != image_id or image.get("file_name") != file_name:
        return None
    return prediction


def _boxes_to_annotations(
    boxes: Any,
    scores: Any,
    image_id: int,
    annotation_id_start: int,
) -> list[dict[str, Any]]:
    annotations = []
    annotation_id = annotation_id_start
    boxes = boxes.detach().float().cpu().tolist()
    scores = scores.detach().float().cpu().tolist()
    for box, score in zip(boxes, scores):
        x_min, y_min, x_max, y_max = (float(value) for value in box)
        width = max(0.0, x_max - x_min)
        height = max(0.0, y_max - y_min)
        if width <= 0 or height <= 0:
            continue
        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [x_min, y_min, width, height],
            "area": width * height,
            "score": float(score),
            "iscrowd": 0,
        })
        annotation_id += 1
    return annotations


class GroundingDINO:
    """Run Transformers Grounding DINO without a segmentation model."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        prompt: str = DEFAULT_PROMPT,
        box_threshold: float = DEFAULT_BOX_THRESHOLD,
        text_threshold: float = DEFAULT_TEXT_THRESHOLD,
        device: str = "auto",
    ) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.prompt = prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()

    def predict(self, image_path: str | Path) -> tuple[Any, Any]:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(
            images=image,
            text=self.prompt,
            return_tensors="pt",
        ).to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        postprocess = self.processor.post_process_grounded_object_detection
        threshold_name = (
            "threshold"
            if "threshold" in inspect.signature(postprocess).parameters
            else "box_threshold"
        )
        results = postprocess(
            outputs,
            inputs.input_ids,
            **{
                threshold_name: self.box_threshold,
                "text_threshold": self.text_threshold,
                "target_sizes": [image.size[::-1]],
            },
        )[0]
        return results["boxes"], results["scores"]


def generate_coco_annotations(
    image_paths: Iterable[str | Path],
    output_json: str | Path,
    predictor: GroundingDINO,
    *,
    prediction_dir: str | Path | None = None,
    resume: bool = False,
    batch_size: int = 1,
) -> None:
    """Generate resumable bbox-only COCO annotations."""
    image_paths = [Path(image_path) for image_path in image_paths]
    output_json = Path(output_json)
    prediction_dir = Path(prediction_dir or output_json.parent / f"{output_json.stem}_predictions")
    prediction_dir.mkdir(parents=True, exist_ok=True)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    for batch_start in tqdm(range(0, len(image_paths), batch_size), desc="Grounding DINO"):
        batch_paths = image_paths[batch_start : batch_start + batch_size]
        for batch_offset, image_path in enumerate(batch_paths):
            image_id = batch_start + batch_offset
            checkpoint_path = prediction_dir / f"{image_id:08d}.json"
            if resume and _load_prediction(checkpoint_path, image_id, image_path.name) is not None:
                continue
            with Image.open(image_path) as image:
                width, height = image.size
            boxes, scores = predictor.predict(image_path)
            annotations = _boxes_to_annotations(boxes, scores, image_id, 1)
            _write_json_atomic(
                checkpoint_path,
                {
                    "image": {
                        "id": image_id,
                        "file_name": image_path.name,
                        "width": width,
                        "height": height,
                    },
                    "annotations": annotations,
                },
            )

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "crater"}],
    }
    annotation_id = 1
    for image_id, image_path in enumerate(image_paths):
        prediction = _load_prediction(
            prediction_dir / f"{image_id:08d}.json",
            image_id,
            image_path.name,
        )
        if prediction is None:
            raise RuntimeError(f"Missing or invalid prediction checkpoint for {image_path}")
        coco["images"].append(prediction["image"])
        for annotation in prediction["annotations"]:
            annotation["id"] = annotation_id
            coco["annotations"].append(annotation)
            annotation_id += 1
    _write_json_atomic(output_json, coco)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image-list", type=Path)
    input_group.add_argument("--image-root", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--box-threshold", type=float, default=DEFAULT_BOX_THRESHOLD)
    parser.add_argument("--text-threshold", type=float, default=DEFAULT_TEXT_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.box_threshold <= 1.0:
        parser.error("--box-threshold must be between 0 and 1")
    if not 0.0 <= args.text_threshold <= 1.0:
        parser.error("--text-threshold must be between 0 and 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    if args.image_list is not None:
        image_paths = _read_image_list(args.image_list, args.dataset_root.resolve())
    else:
        image_paths = _discover_images(args.image_root.resolve())
    if not image_paths:
        parser.error("no corrected images found")
    if args.limit is not None:
        image_paths = image_paths[:args.limit]

    predictor = GroundingDINO(
        args.model_id,
        prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
    )
    generate_coco_annotations(
        image_paths,
        args.output_json,
        predictor,
        prediction_dir=args.prediction_dir,
        resume=args.resume,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
