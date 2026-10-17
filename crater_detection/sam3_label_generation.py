"""Generate COCO instance annotations with SAM 3 concept segmentation."""

from __future__ import annotations

import argparse
import inspect
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Sequence

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


def _discover_images(image_root: Path) -> list[Path]:
    return sorted(image_root.glob("*_corrected.png"))


def _as_mask_array(masks: Any) -> np.ndarray:
    masks = masks.detach().float().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape [N, H, W], got {masks.shape}")
    return masks.astype(bool)


def _as_box_array(boxes: Any) -> np.ndarray:
    boxes = boxes.detach().float().cpu().numpy() if hasattr(boxes, "detach") else np.asarray(boxes)
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Expected boxes with shape [N, 4], got {boxes.shape}")
    return boxes


def _as_score_array(scores: Any) -> np.ndarray:
    scores = scores.detach().float().cpu().numpy() if hasattr(scores, "detach") else np.asarray(scores)
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


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    temporary_path.replace(path)


def _load_prediction(path: Path, image_id: int, file_name: str) -> dict | None:
    try:
        with path.open(encoding="utf-8") as handle:
            prediction = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    image = prediction.get("image", {})
    if image.get("id") != image_id or image.get("file_name") != file_name:
        return None
    if not isinstance(prediction.get("annotations"), list):
        return None
    return prediction


class SAM3Predictor:
    """Load SAM 3 once and run text concept prompts on image batches."""

    def __init__(
        self,
        *,
        device: str = "auto",
        bpe_path: str | Path | None = None,
        detection_threshold: float = 0.5,
    ):
        import torch
        import sam3
        from sam3 import build_sam3_image_model
        from sam3.eval.postprocessors import PostProcessImage
        from sam3.train.data.collator import collate_fn_api
        from sam3.train.transforms.basic_for_api import (
            ComposeAPI,
            NormalizeAPI,
            RandomResizeAPI,
            ToTensorAPI,
        )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        self.torch = torch
        self.device = device
        package_root = Path(sam3.__file__).resolve().parent
        if bpe_path is None:
            bpe_path = package_root / "assets" / "bpe_simple_vocab_16e6.txt.gz"
        bpe_path = Path(bpe_path)

        builder_parameters = inspect.signature(build_sam3_image_model).parameters
        if "bpe_path" in builder_parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in builder_parameters.values()
        ):
            self.model = build_sam3_image_model(bpe_path=str(bpe_path))
        else:
            self.model = build_sam3_image_model()
        self.model = self.model.to(device).eval()
        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=1008,
                    max_size=1008,
                    square=True,
                    consistent_transform=False,
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.collate = collate_fn_api
        self.postprocessor = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=detection_threshold,
            to_cpu=True,
        )

    def _autocast_context(self):
        if self.device == "cuda":
            return self.torch.autocast("cuda", dtype=self.torch.bfloat16)
        return nullcontext()

    @staticmethod
    def _datapoint(image: Image.Image, prompt: str, query_id: int) -> Any:
        from sam3.train.data.sam3_image_dataset import (
            Datapoint,
            FindQueryLoaded,
            Image as SAMImage,
            InferenceMetadata,
        )

        width, height = image.size
        datapoint = Datapoint(find_queries=[], images=[])
        datapoint.images = [SAMImage(data=image, objects=[], size=[height, width])]
        datapoint.find_queries.append(
            FindQueryLoaded(
                query_text=prompt,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=query_id,
                    original_image_id=query_id,
                    original_category_id=1,
                    original_size=[width, height],
                    object_id=0,
                    frame_index=0,
                ),
            )
        )
        return datapoint

    def predict_batch(
        self,
        image_paths: Sequence[str | Path],
        prompt: str,
    ) -> dict[int, dict[str, Any]]:
        """Return post-processed results keyed by image position."""
        from sam3.model.utils.misc import copy_data_to_device

        datapoints = []
        for query_id, image_path in enumerate(image_paths):
            image = Image.open(image_path).convert("RGB")
            datapoint = self._datapoint(image, prompt, query_id)
            datapoints.append(self.transform(datapoint))

        if not datapoints:
            return {}

        batch = self.collate(datapoints, dict_key="mars")["mars"]
        batch = copy_data_to_device(
            batch,
            self.torch.device(self.device),
            non_blocking=self.device == "cuda",
        )
        with self.torch.inference_mode(), self._autocast_context():
            output = self.model(batch)
        processed = self.postprocessor.process_results(output, batch.find_metadatas)
        return {query_id: processed[query_id] for query_id in range(len(image_paths))}

    def predict(self, image_path: str | Path, prompt: str) -> tuple[Any, Any, Any]:
        """Return SAM 3 masks, boxes, and scores for one image."""
        result = self.predict_batch([image_path], prompt)[0]
        return result["masks"], result["boxes"], result["scores"]


def generate_coco_annotations(
    image_paths: Iterable[str | Path],
    output_json: str | Path,
    predictor: SAM3Predictor,
    *,
    prompt: str,
    score_threshold: float = 0.0,
    batch_size: int = 4,
    prediction_dir: str | Path | None = None,
    resume: bool = False,
) -> None:
    """Generate raw SAM 3 predictions with resumable per-image checkpoints."""
    from pycocotools import mask as mask_utils

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    image_paths = [Path(image_path) for image_path in image_paths]
    output_json = Path(output_json)
    if prediction_dir is None:
        prediction_dir = output_json.parent / f"{output_json.stem}_predictions"
    prediction_dir = Path(prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for batch_start in tqdm(range(0, len(image_paths), batch_size), desc="SAM 3"):
        batch_paths = image_paths[batch_start : batch_start + batch_size]
        pending_paths = []
        pending_offsets = []
        for batch_offset, image_path in enumerate(batch_paths):
            image_id = batch_start + batch_offset
            checkpoint_path = prediction_dir / f"{image_id:08d}.json"
            if resume and _load_prediction(checkpoint_path, image_id, image_path.name) is not None:
                continue
            pending_paths.append(image_path)
            pending_offsets.append(batch_offset)
        batch_results = (
            predictor.predict_batch(pending_paths, prompt) if pending_paths else {}
        )
        for pending_offset, batch_offset in enumerate(pending_offsets):
            image_path = batch_paths[batch_offset]
            image_id = batch_start + batch_offset
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")
            height, width = image.shape[:2]
            result = batch_results[pending_offset]
            mask_array = _as_mask_array(result["masks"])
            box_array = _as_box_array(result["boxes"])
            score_array = _as_score_array(result["scores"])
            if not (len(mask_array) == len(box_array) == len(score_array)):
                raise ValueError("SAM 3 returned different numbers of masks, boxes, and scores")
            annotations = []
            for local_annotation_id, (mask, box, score) in enumerate(
                zip(mask_array, box_array, score_array), start=1
            ):
                if score < score_threshold:
                    continue
                annotation = _mask_annotation(
                    mask, image_id, local_annotation_id, score, mask_utils
                )
                if annotation is not None:
                    annotation["sam3_box"] = [float(value) for value in box]
                    annotations.append(annotation)
            _write_json_atomic(
                prediction_dir / f"{image_id:08d}.json",
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
            prediction_dir / f"{image_id:08d}.json", image_id, image_path.name
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
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    image_source = parser.add_mutually_exclusive_group(required=True)
    image_source.add_argument("--image-list", type=Path)
    image_source.add_argument("--image-root", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--prompt", default="circle")
    parser.add_argument("--detection-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bpe-path", type=Path)
    parser.add_argument("--prediction-dir", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    if args.image_list is not None:
        image_paths = _read_image_list(args.image_list, args.dataset_root.resolve())
    else:
        image_paths = _discover_images(args.image_root.resolve())
    if not image_paths:
        parser.error("no corrected images found")
    if args.limit is not None:
        image_paths = image_paths[:args.limit]
    if not 0.0 <= args.detection_threshold <= 1.0:
        parser.error("--detection-threshold must be between 0 and 1")
    if not 0.0 <= args.score_threshold <= 1.0:
        parser.error("--score-threshold must be between 0 and 1")
    predictor = SAM3Predictor(
        device=args.device,
        bpe_path=args.bpe_path,
        detection_threshold=args.detection_threshold,
    )
    generate_coco_annotations(
        image_paths,
        args.output_json,
        predictor,
        prompt=args.prompt,
        score_threshold=args.score_threshold,
        batch_size=args.batch_size,
        prediction_dir=args.prediction_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()