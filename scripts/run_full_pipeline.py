"""Run label generation followed by iterative detector training."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(command: list[str], *, cwd: Path, dry_run: bool) -> None:
    print("$ " + " ".join(command))
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--image-list", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-weights", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--grounding-model-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam-model-type", default="vit_h")
    parser.add_argument("--prompt", default="circle")
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--fbwr-threshold", type=float, default=2000.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_root = args.output_root.resolve()
    labels_dir = output_root / "labels"
    filtered_json = labels_dir / "fbwr_filtered.json"
    scripts_dir = repo_root / "scripts"

    label_command = [
        args.python_executable,
        str(scripts_dir / "run_label_generation.py"),
        "--dataset-root", str(args.dataset_root.resolve()),
        "--image-root", str(args.image_root.resolve()),
        "--image-list", str(args.image_list.resolve()),
        "--sam-checkpoint", str(args.sam_checkpoint.resolve()),
        "--output-dir", str(labels_dir),
        "--grounding-model-id", args.grounding_model_id,
        "--sam-model-type", args.sam_model_type,
        "--prompt", args.prompt,
        "--box-threshold", str(args.box_threshold),
        "--text-threshold", str(args.text_threshold),
        "--fbwr-threshold", str(args.fbwr_threshold),
        "--device", args.device,
    ]
    if args.dry_run:
        label_command.append("--dry-run")
    run(label_command, cwd=repo_root, dry_run=args.dry_run)

    training_command = [
        args.python_executable,
        str(scripts_dir / "run_training_pipeline.py"),
        "--dataset-root", str(args.dataset_root.resolve()),
        "--initial-annotations", str(filtered_json),
        "--initial-weights", str(args.initial_weights.resolve()),
        "--annotations-dir", str(args.annotations_dir.resolve()),
        "--output-root", str(output_root / "training"),
        "--num-gpus", str(args.num_gpus),
        "--python-executable", args.python_executable,
    ]
    if args.dry_run:
        training_command.append("--dry-run")
    run(training_command, cwd=repo_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
