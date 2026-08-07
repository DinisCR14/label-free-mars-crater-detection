"""Run Grounded SAM, geometric refinement, and FBWR filtering in sequence."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


class LabelGenerationPipeline:
    """Coordinate the three label-generation stages and preserve their outputs."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo_root = Path(__file__).resolve().parents[1]
        self.package_dir = self.repo_root / "crater_detection"
        self.output_dir = args.output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.python_executable = args.python_executable

    def _run(self, command: list[str]) -> None:
        print("$ " + " ".join(command))
        if not self.args.dry_run:
            subprocess.run(command, cwd=self.script_dir.parent, check=True)

    def run(self) -> None:
        grounded_sam_json = self.output_dir / "grounded_sam.json"
        refined_json = self.output_dir / "refined.json"
        filtered_json = self.output_dir / "fbwr_filtered.json"

        self._run([
            self.python_executable,
            str(self.package_dir / "grounded_sam.py"),
            "--dataset-root",
            str(self.args.dataset_root.resolve()),
            "--image-list",
            str(self.args.image_list.resolve()),
            "--output-json",
            str(grounded_sam_json),
            "--sam-checkpoint",
            str(self.args.sam_checkpoint.resolve()),
            "--grounding-model-id",
            self.args.grounding_model_id,
            "--sam-model-type",
            self.args.sam_model_type,
            "--prompt",
            self.args.prompt,
            "--box-threshold",
            str(self.args.box_threshold),
            "--text-threshold",
            str(self.args.text_threshold),
            "--device",
            self.args.device,
        ])
        self._run([
            self.python_executable,
            str(self.package_dir / "geometric_boundary_refinement.py"),
            str(grounded_sam_json),
            str(refined_json),
        ])
        self._run([
            self.python_executable,
            str(self.package_dir / "filter_coco_fbwr.py"),
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
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--image-list", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grounding-model-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--sam-model-type", default="vit_h")
    parser.add_argument("--prompt", default="circle")
    parser.add_argument("--box-threshold", type=float, default=0.30)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--fbwr-threshold", type=float, default=2000.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    LabelGenerationPipeline(args).run()


if __name__ == "__main__":
    main()
