"""Run iterative self-training rounds followed by standard-loss training."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    cutler_root: Path
    dataset_root: Path
    train_net: Path
    base_config: Path
    self_train_config: Path
    standard_loss_config: Path
    initial_annotations: Path
    initial_weights: Path
    annotations_dir: Path
    output_root: Path
    train_dataset_prefix: str = "crater_dataset_train_r"
    prediction_dataset: str = "crater_dataset_train"
    thresholds: tuple[float, ...] = (0.70, 0.65, 0.60)
    num_gpus: int = 1
    detections_per_image: int = 30
    python_executable: str = "python"


class TrainingPipeline:
    """Coordinate model inference, label merging, and successive training runs."""

    def __init__(self, config: PipelineConfig, *, dry_run: bool = False):
        self.config = config
        self.python_executable = self._resolve_python_executable(config.python_executable)
        self.dry_run = dry_run
        self.merge_script = (
            Path(__file__).resolve().parents[1] / "crater_detection" / "merge_self_training.py"
        )
        if not dry_run:
            self.config.annotations_dir.mkdir(parents=True, exist_ok=True)
            self.config.output_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_python_executable(executable: str) -> str:
        executable_path = Path(executable).expanduser()
        if executable_path.is_absolute() or executable_path.parent != Path("."):
            resolved = executable_path.resolve()
        else:
            located = shutil.which(executable)
            if located is None:
                raise FileNotFoundError(f"Python executable not found: {executable}")
            resolved = Path(located).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Python executable not found: {resolved}")
        return str(resolved)

    def _run(self, command: list[str]) -> None:
        print("$ " + " ".join(command))
        if not self.dry_run:
            environment = os.environ.copy()
            python_path = [str(self.config.cutler_root), environment.get("PYTHONPATH", "")]
            environment["PYTHONPATH"] = os.pathsep.join(path for path in python_path if path)
            environment["CRATER_DATASET_ROOT"] = str(self.config.dataset_root)
            environment["CRATER_ANNOTATIONS_DIR"] = str(self.config.annotations_dir)
            environment["CRATER_INITIAL_ANNOTATIONS"] = str(self.config.initial_annotations)
            subprocess.run(command, cwd=self.config.cutler_root, check=True, env=environment)

    def _train_net_command(
        self,
        config_file: Path,
        output_dir: Path,
        weights: Path,
        dataset_name: str | None = None,
        eval_only: bool = False,
    ) -> list[str]:
        command = [
            self.python_executable,
            str(self.config.train_net),
            "--num-gpus",
            str(self.config.num_gpus),
            "--config-file",
            str(config_file),
        ]
        if eval_only:
            command.extend([
                "--test-dataset",
                dataset_name or self.config.prediction_dataset,
                "--eval-only",
                "TEST.DETECTIONS_PER_IMAGE",
                str(self.config.detections_per_image),
            ])
        elif dataset_name is not None:
            command.extend(["--train-dataset", dataset_name])
        command.extend([
            "MODEL.WEIGHTS",
            str(weights),
            "OUTPUT_DIR",
            str(output_dir),
        ])
        return command

    def run(self) -> None:
        initial_dir = self.config.output_root / "initial-training"
        self._run(self._train_net_command(
            self.config.base_config,
            initial_dir,
            self.config.initial_weights,
            dataset_name="crater_dataset_train",
        ))
        previous_annotations = self.config.initial_annotations
        previous_weights = initial_dir / "model_final.pth"

        for round_number, threshold in enumerate(self.config.thresholds, start=1):
            inference_dir = self.config.output_root / f"inference-r{round_number}"
            prediction_json = inference_dir / "inference" / "coco_instances_results.json"
            annotation_json = self.config.annotations_dir / f"train_r{round_number}.json"
            train_dir = self.config.output_root / f"self-train-r{round_number}"
            dataset_name = f"{self.config.train_dataset_prefix}{round_number}"

            self._run(self._train_net_command(
                self.config.base_config,
                inference_dir,
                previous_weights,
                eval_only=True,
            ))
            self._run([
                self.python_executable,
                str(self.merge_script),
                "--previous-json",
                str(previous_annotations),
                "--new-predictions-json",
                str(prediction_json),
                "--output-json",
                str(annotation_json),
                "--confidence-threshold",
                str(threshold),
            ])
            self._run(self._train_net_command(
                self.config.self_train_config,
                train_dir,
                previous_weights,
                dataset_name=dataset_name,
            ))
            previous_annotations = annotation_json
            previous_weights = train_dir / "model_final.pth"

        final_dir = self.config.output_root / "final-standard-loss"
        self._run(self._train_net_command(
            self.config.standard_loss_config,
            final_dir,
            previous_weights,
            dataset_name=f"{self.config.train_dataset_prefix}{len(self.config.thresholds)}",
        ))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutler-root", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-net", type=Path)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--self-train-config", type=Path)
    parser.add_argument("--standard-loss-config", type=Path, required=True)
    parser.add_argument("--initial-annotations", type=Path, required=True)
    parser.add_argument("--initial-weights", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cutler_root = (
        args.cutler_root
        or Path(__file__).resolve().parents[1] / "third_party" / "CutLER"
    ).resolve()

    config = PipelineConfig(
        cutler_root=cutler_root,
        dataset_root=args.dataset_root.resolve(),
        train_net=(args.train_net or cutler_root / "cutler" / "train_net.py").resolve(),
        base_config=(
            args.base_config
            or cutler_root
            / "cutler"
            / "model_zoo"
            / "configs"
            / "CutLER-CraterDataset"
            / "cascade_mask_rcnn_R_50_FPN_modified.yaml"
        ).resolve(),
        self_train_config=(
            args.self_train_config
            or cutler_root
            / "cutler"
            / "model_zoo"
            / "configs"
            / "CutLER-CraterDataset"
            / "cascade_mask_rcnn_R_50_FPN_self_train_modified.yaml"
        ).resolve(),
        standard_loss_config=args.standard_loss_config.resolve(),
        initial_annotations=args.initial_annotations.resolve(),
        initial_weights=args.initial_weights.resolve(),
        annotations_dir=args.annotations_dir.resolve(),
        output_root=args.output_root.resolve(),
        num_gpus=args.num_gpus,
        python_executable=args.python_executable,
    )
    TrainingPipeline(config, dry_run=args.dry_run).run()


if __name__ == "__main__":
    main()
