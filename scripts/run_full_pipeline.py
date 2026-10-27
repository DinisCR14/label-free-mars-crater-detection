"""Run mask label generation, dataset preparation, and detector training."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _run(command: list[str], *, cwd: Path, dry_run: bool) -> None:
    print("$ " + " ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("grounded-sam", "sam3"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--initial-weights", type=Path, required=True)
    parser.add_argument("--existing-predictions", type=Path)
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--prompt", default="circle")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--standard-loss-config", type=Path, default=(
        Path(__file__).resolve().parents[1]
        / "third_party/CutLER/cutler/model_zoo/configs/CutLER-CraterDataset"
        / "cascade_mask_rcnn_R_50_FPN_final.yaml"
    ))
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--allow-legacy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.model == "grounded-sam" and args.existing_predictions is None:
        if args.sam_checkpoint is None:
            parser.error("--sam-checkpoint is required for grounded-sam inference")

    repo_root = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    labels_root = output_root / "labels"
    dataset_root = output_root / "dataset"
    training_root = output_root / "training"
    python_args = ["--python-executable", args.python_executable]

    label_command = [
        args.python_executable,
        str(repo_root / "scripts/run_label_generation.py"),
        "--model", args.model,
        "--manifest", str(args.manifest.resolve()),
        "--image-root", str(args.image_root.resolve()),
        "--output-dir", str(labels_root),
        "--prompt", args.prompt,
        "--device", args.device,
    ]
    if args.existing_predictions is not None:
        label_command.extend([
            "--existing-predictions",
            str(args.existing_predictions.resolve()),
        ])
    if args.sam_checkpoint is not None:
        label_command.extend(["--sam-checkpoint", str(args.sam_checkpoint.resolve())])
    if args.allow_legacy:
        label_command.append("--allow-legacy")
    if args.dry_run:
        label_command.append("--dry-run")
    _run(label_command, cwd=repo_root, dry_run=args.dry_run)

    prepare_command = [
        args.python_executable,
        str(repo_root / "tools/dataset/prepare_training_dataset.py"),
        "--manifest", str(args.manifest.resolve()),
        "--image-root", str(args.image_root.resolve()),
        "--pseudo-labels", str((labels_root / "fbwr_filtered.json").resolve()),
        "--ground-truth", str(args.ground_truth.resolve()),
        "--output-root", str(dataset_root),
        "--seed", str(args.seed),
        "--link-mode", args.link_mode,
    ]
    _run(prepare_command, cwd=repo_root, dry_run=args.dry_run)

    training_command = [
        args.python_executable,
        str(repo_root / "scripts/run_training_pipeline.py"),
        "--dataset-root", str(dataset_root),
        "--initial-annotations", str((dataset_root / "annotations/train_initial.json").resolve()),
        "--initial-weights", str(args.initial_weights.resolve()),
        "--annotations-dir", str((dataset_root / "annotations").resolve()),
        "--output-root", str(training_root),
        "--standard-loss-config", str(args.standard_loss_config.resolve()),
        "--num-gpus", str(args.num_gpus),
        *python_args,
    ]
    _run(training_command, cwd=repo_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
