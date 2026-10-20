"""Generate resumable COCO annotations with Grounded SAM."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import cv2
from tqdm import tqdm

from crater_detection.grounded_sam import (
    GroundedSAM,
    GroundedSAMConfig,
    _mask_annotation,
)
from crater_detection.image_manifest import ImageManifest


def _write_json_atomic(path: Path, payload: dict) -> None:
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


def _load_prediction(
    path: Path, image_id: int, file_name: str, manifest: ImageManifest
) -> dict | None:
    try:
        prediction = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    image = prediction.get("image", {})
    if image.get("id") != image_id or image.get("file_name") != file_name:
        return None
    if prediction.get("mapping") != manifest.metadata():
        return None
    if not isinstance(prediction.get("annotations"), list):
        return None
    return prediction


def generate_coco_annotations(
    image_paths: list[Path],
    output_json: str | Path,
    predictor: GroundedSAM,
    *,
    manifest: ImageManifest,
    prediction_dir: str | Path | None = None,
    resume: bool = False,
) -> None:
    """Generate Grounded SAM predictions with resumable checkpoints."""
    from pycocotools import mask as mask_utils

    if len(image_paths) > len(manifest.images):
        raise ValueError("The image path list does not match the manifest")
    output_json = Path(output_json)
    prediction_dir = Path(
        prediction_dir or output_json.parent / f"{output_json.stem}_predictions"
    )
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for image, image_path in tqdm(
        zip(manifest.images, image_paths), total=len(image_paths), desc="Grounded SAM"
    ):
        checkpoint_path = prediction_dir / f"{image.id:08d}.json"
        if resume and _load_prediction(
            checkpoint_path, image.id, image.file_name, manifest
        ) is not None:
            continue
        source = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if source is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        height, width = source.shape[:2]
        if (width, height) != (image.width, image.height):
            raise ValueError(f"Image dimensions do not match manifest: {image_path}")
        masks, _ = predictor.predict(image_path)
        annotations = []
        for annotation_id, mask in enumerate(masks, start=1):
            annotation = _mask_annotation(mask, image.id, annotation_id, mask_utils)
            if annotation is not None:
                annotations.append(annotation)
        _write_json_atomic(
            checkpoint_path,
            {
                "image": {
                    "id": image.id,
                    "file_name": image.file_name,
                    "width": width,
                    "height": height,
                },
                "mapping": manifest.metadata(),
                "annotations": annotations,
            },
        )

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "crater"}],
        "mapping": manifest.metadata(),
    }
    annotation_id = 1
    for image, image_path in zip(manifest.images, image_paths):
        prediction = _load_prediction(
            prediction_dir / f"{image.id:08d}.json",
            image.id,
            image_path.name,
            manifest,
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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path)
    parser.add_argument("--grounding-model-id", default=GroundedSAMConfig.grounding_model_id)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--sam-model-type", default=GroundedSAMConfig.sam_model_type)
    parser.add_argument("--prompt", default=GroundedSAMConfig.prompt)
    parser.add_argument("--box-threshold", type=float, default=GroundedSAMConfig.box_threshold)
    parser.add_argument("--text-threshold", type=float, default=GroundedSAMConfig.text_threshold)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=GroundedSAMConfig.device)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not 0.0 <= args.box_threshold <= 1.0:
        parser.error("--box-threshold must be between 0 and 1")
    if not 0.0 <= args.text_threshold <= 1.0:
        parser.error("--text-threshold must be between 0 and 1")

    try:
        manifest = ImageManifest.load(args.manifest)
        image_paths = manifest.resolve(args.image_root.resolve(), limit=args.limit)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    config = GroundedSAMConfig(
        grounding_model_id=args.grounding_model_id,
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
    )
    generate_coco_annotations(
        image_paths,
        args.output_json,
        GroundedSAM(config),
        manifest=manifest,
        prediction_dir=args.prediction_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
