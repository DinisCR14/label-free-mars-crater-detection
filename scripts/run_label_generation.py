"""Run mask-producing inference and the shared label post-processing stages."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crater_detection.image_manifest import ImageManifest, validate_coco_mapping


SUPPORTED_MODELS = ("grounded-sam", "sam3")


class LabelGenerationPipeline:
    """Dispatch mask inference and apply common refinement and FBWR filtering."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo_root = Path(__file__).resolve().parents[1]
        self.output_dir = args.output_dir.resolve()
        self.raw_dir = self.output_dir / "raw"
        self.raw_json = self.raw_dir / "predictions.json"
        self.manifest = ImageManifest.load(args.manifest)
        if not args.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _run(self, command: list[str]) -> None:
        print("$ " + " ".join(command))
        if not self.args.dry_run:
            subprocess.run(command, cwd=self.repo_root, check=True)

    def _validate_predictions(self, path: Path) -> None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_coco_mapping(
            payload,
            self.manifest,
            require_fingerprint=not self.args.allow_legacy,
        )
        for annotation in payload.get("annotations", []):
            if not isinstance(annotation.get("segmentation"), (list, dict)):
                raise ValueError("Mask-producing predictions require segmentation data")

    def _prepare_raw_predictions(self) -> None:
        if self.args.existing_predictions is not None:
            source = self.args.existing_predictions.resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Prediction file does not exist: {source}")
            self._validate_predictions(source)
            if not self.args.dry_run:
                self.raw_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, self.raw_json)
            return

        runner = (
            self.repo_root / "crater_detection" / "sam3_label_generation.py"
            if self.args.model == "sam3"
            else self.repo_root / "crater_detection" / "grounded_sam_label_generation.py"
        )
        command = [
            self.args.python_executable,
            str(runner),
            "--manifest",
            str(self.args.manifest.resolve()),
            "--image-root",
            str(self.args.image_root.resolve()),
            "--output-json",
            str(self.raw_json),
            "--prediction-dir",
            str((self.raw_dir / "checkpoints").resolve()),
            "--prompt",
            self.args.prompt,
            "--device",
            self.args.device,
        ]
        if self.args.resume:
            command.append("--resume")
        if self.args.limit is not None:
            command.extend(["--limit", str(self.args.limit)])
        if self.args.model == "sam3":
            command.extend([
                "--detection-threshold",
                str(self.args.detection_threshold),
                "--score-threshold",
                str(self.args.score_threshold),
                "--batch-size",
                str(self.args.batch_size),
            ])
            if self.args.bpe_path is not None:
                command.extend(["--bpe-path", str(self.args.bpe_path.resolve())])
        else:
            command.extend([
                "--sam-checkpoint",
                str(self.args.sam_checkpoint.resolve()),
                "--grounding-model-id",
                self.args.grounding_model_id,
                "--sam-model-type",
                self.args.sam_model_type,
                "--box-threshold",
                str(self.args.box_threshold),
                "--text-threshold",
                str(self.args.text_threshold),
            ])
        self._run(command)

    def run(self) -> None:
        self._prepare_raw_predictions()
        refined_json = self.output_dir / "refined.json"
        filtered_json = self.output_dir / "fbwr_filtered.json"
        self._run([
            self.args.python_executable,
            str(self.repo_root / "crater_detection" / "geometric_boundary_refinement.py"),
            str(self.raw_json),
            str(refined_json),
        ])
        self._run([
            self.args.python_executable,
            str(self.repo_root / "crater_detection" / "filter_coco_fbwr.py"),
            str(refined_json),
            str(self.args.image_root.resolve()),
            str(filtered_json),
            "--threshold",
            str(self.args.fbwr_threshold),
            "--device",
            self.args.device,
        ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--existing-predictions", type=Path)
    parser.add_argument("--allow-legacy", action="store_true")
    parser.add_argument("--prompt", default="circle")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fbwr-threshold", type=float, default=2000.0)
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--grounding-model-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam-model-type", default="vit_h")
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--detection-threshold", type=float, default=0.50)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bpe-path", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    for name in (
        "box_threshold",
        "text_threshold",
        "detection_threshold",
        "score_threshold",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.existing_predictions is None and args.model == "grounded-sam":
        if args.sam_checkpoint is None:
            parser.error("--sam-checkpoint is required for grounded-sam inference")
    try:
        LabelGenerationPipeline(args).run()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
