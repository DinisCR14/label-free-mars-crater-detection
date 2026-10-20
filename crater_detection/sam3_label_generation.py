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

from crater_detection.image_manifest import ImageManifest, ManifestImage


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


def _thresholded_coco_document(
    coco: dict,
    score_threshold: float,
) -> dict:
    output = {
        key: value
        for key, value in coco.items()
        if key not in {"annotations", "score_threshold"}
    }
    output["score_threshold"] = score_threshold
    output["annotations"] = [
        annotation
        for annotation in coco["annotations"]
        if annotation.get("score", 0.0) >= score_threshold
    ]
    return output


def _load_prediction(
    path: Path,
    image_id: int,
    file_name: str,
    manifest: ImageManifest | None = None,
) -> dict | None:
    try:
        with path.open(encoding="utf-8") as handle:
            prediction = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    image = prediction.get("image", {})
    if image.get("id") != image_id or image.get("file_name") != file_name:
        return None
    if manifest is not None and prediction.get("mapping") != manifest.metadata():
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
    manifest: ImageManifest | None = None,
) -> None:
    """Generate raw SAM 3 predictions with resumable per-image checkpoints."""
    from pycocotools import mask as mask_utils

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    image_paths = [Path(image_path) for image_path in image_paths]
    records = (
        list(manifest.images[: len(image_paths)])
        if manifest is not None
        else [ManifestImage(index, path.name, 0, 0) for index, path in enumerate(image_paths)]
    )
    if len(records) != len(image_paths):
        raise ValueError("The image path list does not match the manifest")
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
            image_id = records[batch_start + batch_offset].id
            checkpoint_path = prediction_dir / f"{image_id:08d}.json"
            if (
                resume
                and _load_prediction(
                    checkpoint_path, image_id, image_path.name, manifest
                )
                is not None
            ):
                continue
            pending_paths.append(image_path)
            pending_offsets.append(batch_offset)
        batch_results = (
            predictor.predict_batch(pending_paths, prompt) if pending_paths else {}
        )
        for pending_offset, batch_offset in enumerate(pending_offsets):
            image_path = batch_paths[batch_offset]
            image_id = records[batch_start + batch_offset].id
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
                    "mapping": manifest.metadata() if manifest is not None else None,
                    "annotations": annotations,
                },
            )

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "crater"}],
    }
    if manifest is not None:
        coco["mapping"] = manifest.metadata()
    annotation_id = 1
    for record, image_path in zip(records, image_paths):
        image_id = record.id
        prediction = _load_prediction(
            prediction_dir / f"{image_id:08d}.json",
            image_id,
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


def generate_score_threshold_sweep(
    image_paths: Iterable[str | Path],
    output_dir: str | Path,
    predictor: SAM3Predictor,
    *,
    prompt: str,
    score_thresholds: Iterable[float],
    batch_size: int = 4,
    prediction_dir: str | Path | None = None,
    resume: bool = False,
    manifest: ImageManifest | None = None,
) -> None:
    """Run SAM 3 once and write one COCO document per score threshold."""
    thresholds = sorted({float(value) for value in score_thresholds})
    if not thresholds or any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("score_thresholds must contain values between 0 and 1")

    output_dir = Path(output_dir)
    raw_output = output_dir / "sam3-all-scores.json"
    generate_coco_annotations(
        image_paths,
        raw_output,
        predictor,
        prompt=prompt,
        score_threshold=0.0,
        batch_size=batch_size,
        prediction_dir=prediction_dir or output_dir / "all-scores_predictions",
        resume=resume,
        manifest=manifest,
    )
    raw_coco = json.loads(raw_output.read_text(encoding="utf-8"))
    for threshold in thresholds:
        threshold_name = f"score-{threshold:.2f}".replace(".", "p")
        threshold_output = output_dir / threshold_name / f"sam3-{prompt}.json"
        _write_json_atomic(
            threshold_output,
            _thresholded_coco_document(raw_coco, threshold),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--prompt", default="circle")
    parser.add_argument("--detection-threshold", type=float, default=0.5)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument(
        "--score-thresholds",
        type=str,
        help="Comma-separated thresholds for a single-inference score sweep",
    )
    parser.add_argument(
        "--threshold-output-dir",
        type=Path,
        help="Directory for score-sweep outputs",
    )
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

    try:
        manifest = ImageManifest.load(args.manifest)
        image_paths = manifest.resolve(args.image_root.resolve(), limit=args.limit)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    if not 0.0 <= args.detection_threshold <= 1.0:
        parser.error("--detection-threshold must be between 0 and 1")
    if not 0.0 <= args.score_threshold <= 1.0:
        parser.error("--score-threshold must be between 0 and 1")
    if not args.score_thresholds and args.output_json is None:
        parser.error("--output-json is required unless --score-thresholds is used")
    if args.score_thresholds and not args.threshold_output_dir:
        parser.error("--threshold-output-dir is required with --score-thresholds")
    if args.threshold_output_dir and not args.score_thresholds:
        parser.error("--score-thresholds is required with --threshold-output-dir")
    predictor = SAM3Predictor(
        device=args.device,
        bpe_path=args.bpe_path,
        detection_threshold=0.0 if args.score_thresholds else args.detection_threshold,
    )
    if args.score_thresholds:
        try:
            thresholds = [float(value) for value in args.score_thresholds.split(",")]
        except ValueError as error:
            parser.error(f"invalid --score-thresholds: {error}")
        generate_score_threshold_sweep(
            image_paths,
            args.threshold_output_dir,
            predictor,
            prompt=args.prompt,
            score_thresholds=thresholds,
            batch_size=args.batch_size,
            prediction_dir=args.prediction_dir,
            resume=args.resume,
            manifest=manifest,
        )
    else:
        generate_coco_annotations(
            image_paths,
            args.output_json,
            predictor,
            prompt=args.prompt,
            score_threshold=args.score_threshold,
            batch_size=args.batch_size,
            prediction_dir=args.prediction_dir,
            resume=args.resume,
            manifest=manifest,
        )


if __name__ == "__main__":
    main()