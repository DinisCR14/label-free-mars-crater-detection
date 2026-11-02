import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.run_training_pipeline import PipelineConfig, TrainingPipeline


class TrainingPipelineTests(unittest.TestCase):
    def test_python_executable_preserves_virtual_environment_symlink(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            real_python = root / "real-python"
            virtualenv_python = root / "venv" / "bin" / "python"
            virtualenv_python.parent.mkdir(parents=True)
            real_python.write_text("#!/bin/sh\n")
            virtualenv_python.symlink_to(real_python)

            resolved = TrainingPipeline._resolve_python_executable(
                str(virtualenv_python)
            )

            self.assertEqual(resolved, str(virtualenv_python))
            self.assertTrue(os.path.islink(resolved))

    def test_initial_only_runs_initial_training_only(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config = PipelineConfig(
                cutler_root=root / "cutler",
                dataset_root=root / "dataset",
                train_net=root / "train_net.py",
                base_config=root / "base.yaml",
                self_train_config=root / "self-train.yaml",
                standard_loss_config=root / "final.yaml",
                initial_annotations=root / "annotations" / "train_initial.json",
                initial_weights=root / "weights.pth",
                annotations_dir=root / "annotations",
                output_root=root / "training",
                python_executable=sys.executable,
            )
            pipeline = TrainingPipeline(config, dry_run=True)
            commands: list[list[str]] = []
            pipeline._run = commands.append  # type: ignore[method-assign]

            pipeline.run(initial_only=True)

            self.assertEqual(len(commands), 1)
            self.assertIn("initial-training", commands[0][-1])
            self.assertNotIn("inference-r1", " ".join(commands[0]))
            self.assertNotIn("self-train-r1", " ".join(commands[0]))
            self.assertNotIn("final-standard-loss", " ".join(commands[0]))


if __name__ == "__main__":
    unittest.main()
