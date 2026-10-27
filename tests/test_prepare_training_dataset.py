import json
import tempfile
import unittest
from pathlib import Path

from tools.dataset.prepare_training_dataset import prepare_training_dataset


class PrepareTrainingDatasetTests(unittest.TestCase):
    def test_creates_deterministic_split_and_subsets_documents(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_root = root / "images"
            image_root.mkdir()
            images = []
            for image_id in range(10):
                name = f"image-{image_id}.png"
                (image_root / name).write_bytes(b"image")
                images.append({
                    "id": image_id,
                    "file_name": name,
                    "width": 4,
                    "height": 4,
                })
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps({
                "schema_version": 1,
                "image_count": len(images),
                "images": images,
            }))
            pseudo_path = root / "pseudo.json"
            pseudo_path.write_text(json.dumps({
                "images": images,
                "annotations": [
                    {"id": 1, "image_id": image_id, "bbox": [0, 0, 1, 1]}
                    for image_id in range(10)
                ],
                "categories": [{"id": 1, "name": "crater"}],
            }))
            ground_truth_path = root / "ground-truth.json"
            ground_truth_path.write_text(json.dumps({
                "images": images,
                "annotations": [],
                "categories": [{"id": 1, "name": "crater"}],
            }))

            output = root / "prepared"
            result = prepare_training_dataset(
                manifest_path,
                image_root,
                pseudo_path,
                ground_truth_path,
                output,
                seed=7,
                link_mode="copy",
            )

            self.assertEqual(
                result["images"]["train"],
                prepare_training_dataset(
                    manifest_path,
                    image_root,
                    pseudo_path,
                    ground_truth_path,
                    root / "prepared-again",
                    seed=7,
                    link_mode="copy",
                )["images"]["train"],
            )
            train = json.loads(
                (output / "annotations" / "train_initial.json").read_text()
            )
            self.assertEqual(len(train["images"]), 8)
            self.assertEqual(len(train["annotations"]), 8)
            self.assertEqual(len(list((output / "train").glob("*.png"))), 8)
            self.assertEqual(len(list((output / "val").glob("*.png"))), 1)
            self.assertEqual(len(list((output / "test").glob("*.png"))), 1)


if __name__ == "__main__":
    unittest.main()
